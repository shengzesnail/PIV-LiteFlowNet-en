%% function test_pivLiteflownet_all
% ----------------------------------------------
% This is a script for evaluating 
%   PIV-LiteFlowNet-en on a list of images
% ----------------------------------------------
% To use MATLAB as a Caffe master, you need to 
%   - add Matlab path in Makefile.config file before compiling Caffe
%   - do 'make matcaffer' while compiling
% ----------------------------------------------
% Lisence and Citation
%   The codes are provided for research purposed only. All rights reserved.
%   Any commercial use requires the consent of the authors. If the codes 
%   are used in your research work, please cite the following papers: 
%       Cai S, Liang J, Gao Q, Xu C, Wei R. Particle image 
%       velocimetry based on a deep learning motion estimator[J].
%       submitted to IEEE transactions on instrumentation and measurement.
%   or the predecessor
%       Cai S, Zhou S, Xu C, Gao Q. Dense motion estimation of 
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


%% ------------------------------------------------------------------------
% select a case for evaluating
% In this demo, the tested case is a uniform flow with (u=5, v=0)
data_root = 'testedData2/test_dx_5/';
% load path for the original images and ground-truth
imagePattern = 'uniform_%05d_img%d.tif';
flowPattern = 'uniform_%05d_flow.flo';


dataSize = 10;

%% ------------------------------------------------------------------------
width = -1;
height = -1;
flag_display = false;
flag_compareResult = true;
flag_writeErrorsToFile = false;



%% ------------------------------------------------------------------------
if flag_compareResult==true
    RMSE = zeros(dataSize,1);
end


%% -----------------------------------------------------------------------
for index = 1:dataSize
    
    fprintf(['Processing data: ----- %05d \n'], index );

    %% Loading images
    % -------------------------------------------------------------------
    img1_name = [data_root, sprintf(imagePattern,index,1)];
    img2_name = [data_root, sprintf(imagePattern,index,2)];
    img1 = imread(img1_name); 
    img2 = imread(img2_name);
    img1 = double(img1);
    img2 = double(img2);
    input_data = prepare_input_data(img1, img2);

    tic
    if width~=size(img1,2) || height~=size(img1,1)
        width =  size(img1,2);
        height = size(img1,1);
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

        % read the data from file
        i=0;
        while feof(proto)==0
            i = i+1;
            tline_ori{i} = fgetl(proto);
        end
        line_size = i;
        % write the data to a new file
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
        clear line line_size i

        % initial network
        % ----------------------------------------------------------------
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

    end

    % -------------------------------------------------------------------
    res = net.forward(input_data);
    res = res{1};
    uv = permute(res, [2,1,3]);
    
    % if median filter is used
%     dx = medfilt2(uv(:,:,1),[5,5],'symmetric'); 
%     dy = medfilt2(uv(:,:,2),[5,5],'symmetric'); 
%     uv = cat(3,dx,dy);
    
    Time = toc;
    
    
    % -------------------------------------------------------------------
    if flag_display==true
%         vort = computeCurl(uv);
        figure;
        plotFlow_Cai(uv(:,:,1), uv(:,:,2), [], 1.0);
        title('Estimated Flow Field');
    end
    
    % -------------------------------------------------------------------
    if flag_compareResult==true
        gt_name = [data_root,sprintf(flowPattern,index)];
        uv_gt = readFlowFile(gt_name);
        
        % if you want to compute RMSE, do it here
        % RMSE for uniform flow in this case
        RMSE(index) = sqrt( mean( (uv_gt(:)-uv(:)).^2 ) );
        fprintf(' -----   RMSE = %3.3f, computation time = %3.3f \n'...
            , RMSE(index), Time);
    end
    
    fprintf('\n');
    


end

%% -----------------------------------------------------------------------
if flag_compareResult==true
    mean_RMSE = mean(RMSE);
    fprintf('Mean RMSE of the whole sequence is %3.3f \n', mean_RMSE);
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



