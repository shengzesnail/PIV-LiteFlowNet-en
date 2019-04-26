function plotFlow_Cai(u, v, imgOriginal, scale, rSize)
% Creates a quiver plot that displays the optical flow vectors on the
% original first frame (if provided). See the MATLAB Function Reference for
% "quiver" for more info.
%
% Usage:
% plotFlow(u, v, imgOriginal, scale, rSize)
%
% u and v are the horizontal and vertical optical flow vectors,
% respectively. imgOriginal, if supplied, is the first frame on which the
% flow vectors would be plotted. use an empty matrix '[]' for no image.
% rSize is the size of the region in which one vector is visible. scale
% over-rules the auto scaling.
%
% Author: Mohd Kharbat at Cranfield Defence and Security
% mkharbat(at)ieee(dot)org , http://mohd.kharbat.com
% Published under a Creative Commons Attribution-Non-Commercial-Share Alike
% 3.0 Unported Licence http://creativecommons.org/licenses/by-nc-sa/3.0/
%
% October 2008
% Rev: Jan 2009
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rev: Shengze Cai, March 2016
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ��ʼ��
% figure();
if nargin>2
    if sum(sum(imgOriginal))~=0
        imagesc(imgOriginal);
        load('BuYlRd.mat')
        colormap(BuYlRd);
%         colorbar();
        hold on;
    end
end
if nargin<4
    scale = 1;
end
if nargin<5
    s = size(u);
    rSize = min(s / 40);
end


%%
Y = size(u,1);  
X = size(u,2);  
[x,y] = meshgrid(0.5:rSize:X, 0.5:rSize:Y);
u_plot = interp2(u, x, y);
v_plot = interp2(v, x, y);   
quiver(x,y,u_plot, v_plot, scale, 'color', 'k', 'linewidth', 1);
set(gca,'YDir','reverse');
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
axis image;

