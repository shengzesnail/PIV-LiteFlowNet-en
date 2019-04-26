%% function test_pivLiteflownet
% ----------------------------------------------
% This is a script for evaluating PIV-LiteFlowNet-en on a single image pair
% ----------------------------------------------
% To use MATLAB as a Caffe master, you need to 
%   - add Matlab path in Makefile.config file before compiling Caffe
%   - do 'make matcaffer' while compiling
% ----------------------------------------------
% Lisence and Citation
%   The codes are provided for research purposed only. All rights reserved.
%   Any commercial use requires the consent of the authors. If the codes 
%   are used in your research work, please cite the following papers: 
%     - Cai S, Liang J, Gao Q, Xu C, Wei R. Particle image 
%       velocimetry based on a deep learning motion estimator[J].
%       submitted to IEEE transactions on instrumentation and measurement.
%   or the predecessor
%     - Cai S, Zhou S, Xu C, Gao Q. Dense motion estimation of 
%       particle images via a convolutional neural network[J]. 
%       Experiments in Fluids, 2019, 60(4): 73.
% ----------------------------------------------
% Edited by Shengze Cai, 2019/04
% ----------------------------------------------



%% ----------------------------------------------
clear;
close all;
clc
addpath(genpath('tools'));
% add caffe-matlab path (absolute path is recommended)
addpath(genpath('../PIV-LiteFlowNet-en/caffe/matlab'));


%% set path (absolute path is recommended)
caffe_root = '../PIV-LiteFlowNet-en/caffe/';
% select a model from somewhere 
netType = 'PIV-LiteFlowNet-en';    % PIV-LiteFlowNet or PIV-LiteFlowNet-en
model_def = [caffe_root, 'models/', netType, '/', netType, '_deploy.prototxt'];
model_weights = [caffe_root, 'models/', netType,'/', netType, '.caffemodel'];

model_def_temp = 'temp_deploy.prototxt.template';


%% select a image case for testing: 
%   vortexPair (without true velocity field)
%   DNS_turbulence or backstep_Re1000 (with true velocity field)
image_root = 'testedData/';
imageType =  'vortexPair';
img0_path = [image_root, imageType,'_img1.tif'];
img1_path = [image_root, imageType, '_img2.tif'];
outFlow_path = [image_root, imageType, '-', netType, '.flo'];

flag_groundTruth = false;   % if ground-truth is available or not
flag_display = true;
flag_writeUvToFile = false;


if ~ exist(model_weights,'file'), error(['caffemodel does not exist: ', model_weights]); end
if ~ exist(model_def,'file'), error(['deploy-proto does not exist: ', model_def]); end
if ~ exist(img0_path,'file'), error(['img0 does not exist: ', img0_path]); end
if ~ exist(img1_path,'file'), error(['img1 does not exist: ', img1_path]); end


%% Loading images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_blobs = 2;

img0 = imread(img0_path); 
img1 = imread(img1_path);
img0 = double(img0);
img1 = double(img1);
input_data = prepare_input_data(img0, img1);
% figure; imshow(img0,[]);

%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% preparing the prototxt file
width = size(img0,2);
height = size(img0,1);
vars = {'$TARGET_WIDTH', '$TARGET_HEIGHT', '$ADAPTED_WIDTH',...
    '$ADAPTED_HEIGHT', '$SCALE_WIDTH', '$SCALE_HEIGHT'};
divisor = 32;
values(1) = width;      values(2) = height;
values(3) = floor(width/divisor) * divisor;
values(4) = floor(height/divisor) * divisor;
values(5) = width/single(values(3));
values(6) = height/single(values(4));

proto = fopen(model_def, 'r'); 
tmp = fopen(model_def_temp,'w');

% read the data from prototxt file
i=0;
while feof(proto)==0
    i = i+1;
    tline_ori{i} = fgetl(proto);
end
line_size = i;
% write the data to a temporary prototxt file
for i = 1:line_size
    line = tline_ori{i};
    for j = 1:length(values)
        line = strrep(line, vars{j}, num2str(values(j)) );
    end
    fprintf(tmp, '%s\n', line);
end
% close the files
fclose(proto);
fclose(tmp);


tic
%% network initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% caffe.set_logging_disabled();
use_gpu = 1;
if use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end
net = caffe.Net(model_def_temp, model_weights, 'test');

Time_loadModel = toc;

%% Network forward computing


res_all = net.forward(input_data);
res = res_all{1};
uv = permute(res, [2,1,3]);

% if median filter is applied
% u = medfilt2(uv(:,:,1),[5,5],'symmetric'); 
% v = medfilt2(uv(:,:,2),[5,5],'symmetric');
% uv = cat(3,u,v);
time_forward = toc;



%% plot the velocity field
if flag_display==true
    figure('color',[1,1,1]);
    subplot(121); imshow(img0,[]);  title('First frame')
    subplot(122); imshow(img1,[]);  title('Second frame')
%     mag = sqrt( uv(:,:,1).^2 + uv(:,:,2).^2 ); 
%     vort = computeCurl(uv); 
    figure('color',[1,1,1]);
    plotFlow_Cai(uv(:,:,1), uv(:,:,2), [], 1.0);
    title(['Estimated Flow Field of ', netType]);
end

%% write the flow field to a .flo file
if flag_writeUvToFile==true
    writeFlowFile(uv, outFlow_path);
end

%% load the ground-truth
if flag_groundTruth==true
    gt_path = [image_root, imageType,'_gt.flo'];
    uv_gt = readFlowFile(gt_path);
    
    % compute the velocity magnitude
    mag_gt = sqrt( uv_gt(:,:,1).^2 + uv_gt(:,:,2).^2 ); 
    % compute the vorticity
    % vort_gt = computeCurl(uv_gt);  
    
    figure('color',[1,1,1]);
    plotFlow_Cai(uv_gt(:,:,1), uv_gt(:,:,2), [], 1.0);
    title('Ground-truth velocity field');  
    
    % if you want to compute the RMSE, do it here
    %   
end




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------------------------------------------------------------
function input_data = prepare_input_data(img0, img1)
% ------------------------------------------------------------------------
    input_data = {};
    if size(size(img0),2)<3
        im = single(img0);
        im = permute(im, [2,1]);
        im_data(:,:,:,1) = cat(3,im,im,im);
        input_data{end+1} = im_data;
    else 
        im = single(img0);
        im_data = permute(im, [2,1,3]);
        input_data{end+1} = im_data;
    end
    if size(size(img1),2)<3
        im = single(img1);
        im = permute(im, [2,1]);
        im_data(:,:,:,1) = cat(3,im,im,im);
        input_data{end+1} = im_data;
    else
        im = single(img1);
        im_data = permute(im, [2,1,3]);
        input_data{end+1} = im_data;
    end

end


