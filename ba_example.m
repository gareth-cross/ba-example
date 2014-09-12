% ba_example.m
% Simple example of bundle adjustment.
% 11/09/2014
% Refer to "A tutorial on SE(3) transformation parameterizations
% and on-manifold optimization" - J.L. Blanco, for explanation of Jacobians

close all;
clear;
home;

NPOSES = 4; % fix this for now
wRb_cams = zeros(3,3,NPOSES);
p_cams = zeros(3,1,NPOSES);

% input 4 initial poses (add more here and increment NPOSES appropriately)
wRb_cams(:,:,1) = rot_x(-pi/2);
p_cams(:,:,1) = zeros(3,1);
wRb_cams(:,:,2) = rot_z(0.4) * wRb_cams(:,:,1);
p_cams(:,:,2) = [1.0; 0; 0];
wRb_cams(:,:,3) = rot_z(0.1) * wRb_cams(:,:,2);
p_cams(:,:,3) = [1.3; 0; 0];
wRb_cams(:,:,4) = rot_z(-0.5) * wRb_cams(:,:,1);
p_cams(:,:,4) = [-1.3; 0; 0];

% generate noisy initial guess poses
ROTATION_NOISE_STD = 0.5/180 * pi;
POSITION_NOISE_STD = 0.7;

wRb_cams_noisy = zeros(3,3,NPOSES);
p_cams_noisy = zeros(3,1,NPOSES);

for j=1:NPOSES
    noise_scale = max((j-2),0) / (NPOSES-2);
    angs = noise_scale*ROTATION_NOISE_STD*randn(3,1);
    noise_rot = rot_x(angs(1)) * rot_y(angs(2)) * rot_z(angs(3));
    noise_pos = noise_scale*POSITION_NOISE_STD*randn(3,1);
    wRb_cams_noisy(:,:,j) = noise_rot * wRb_cams(:,:,j);
    p_cams_noisy(:,:,j) = noise_pos + p_cams(:,:,j);
end

% generate point cloud
NPTS = 50;
point_center = [0; 4.0; 0];
point_rad = 1;
point_std = [0.01; 0.01; 0.01];
points_world = zeros(3,NPTS);

for i=1:NPTS
    R = rot_y(i / NPTS * 2 * pi) * rot_z(i / NPTS * pi/3);
    rad = 0.5 + point_rad*(i / NPTS);
    point = R * [rad; 0; 0];
    points_world(:,i) = point_center + point + point_std .* randn(3,1);
end

% plot point cloud
f3d = figure;
hold on;
scatter3(points_world(1,:), points_world(2,:), points_world(3,:), 'b');
title('Simulated 3D point cloud');
grid on;
axis equal;
axis vis3d;

% plot noisy camera SRTs
for j=1:NPOSES
    wRb = wRb_cams_noisy(:,:,j);
    cPo = p_cams_noisy(:,:,j);
    
    zcam = wRb * [0;0;1];
    xcam = wRb * [1;0;0];
    ycam = wRb * [0;1;0];

    % camera vector
    h = quiver3(cPo(1),cPo(2),cPo(3),zcam(1)*0.5,zcam(2)*0.5,zcam(3)*0.5,'m');
    set(h,'linewidth',2);
    h = quiver3(cPo(1),cPo(2),cPo(3),xcam(1)*0.5,xcam(2)*0.5,xcam(3)*0.5,'k');
    set(h,'linewidth',2);
    h = quiver3(cPo(1),cPo(2),cPo(3),ycam(1)*0.5,ycam(2)*0.5,ycam(3)*0.5,'y');
    set(h,'linewidth',2);
end

% project points into images
points_image = zeros(3,NPTS,NPOSES);
points_image_noisy = zeros(3,NPTS,NPOSES);
points_image_noisy(3,:,:) = 1;

% std deviation on image noise
FOCAL_LENGTH = 500;
IMAGE_NOISE_STD = 0.3 / FOCAL_LENGTH;

OUTLIER_PROB = 0.1;    % probability of a _bad_ outlier
OUTLIER_IMAGE_NOISE_STD = 30 / FOCAL_LENGTH;

% binomial distribution on outliers
binomial = makedist('Binomial', 'N', 1, 'p', OUTLIER_PROB);

f2d = figure;
total_outliers = 0;
for j=1:NPOSES
    wRb = wRb_cams(:,:,j);
    p = p_cams(:,:,j);
    points_image(:,:,j) = wRb' * bsxfun(@minus,points_world,p);
    % divide by camera z coordinate
    points_image(:,:,j) = bsxfun(@rdivide, points_image(:,:,j), points_image(3,:,j));
    
    % add synthetic noise on all features
    points_image_noisy(1:2,:,j) = points_image(1:2,:,j) + IMAGE_NOISE_STD*randn(2,NPTS);
    
    % generate indices of outliers
    outlier_idx = logical(random(binomial, 1, NPTS));
    total_outliers = total_outliers + nnz(outlier_idx);
    points_image_noisy(1:2,outlier_idx,j) = points_image(1:2,outlier_idx,j) + OUTLIER_IMAGE_NOISE_STD*randn(2,nnz(outlier_idx));
    
    % plot resulting points
    subplot(NPOSES,1,j);
    hold on;
    scatter(points_image(1,:,j), points_image(2,:,j), 'b');
    scatter(points_image_noisy(1,:,j), points_image_noisy(2,:,j), 'r');
end
fprintf('Total number of outliers: %i\n', total_outliers);

% estimated poses
wRb_cams_estimate = wRb_cams_noisy;
p_cams_estimate = p_cams_noisy;

% triangulate initial guesses on all features w/ least squares
points_world_estimate = zeros(3,1,NPTS);
for i=1:NPTS
    A = zeros(3,3);
    b = zeros(3,1);
    
    % all observations of this feature, normalized
    u = squeeze( points_image_noisy(:,i,:) );
    u = bsxfun(@rdivide, u, sqrt(sum(u.^2,1)));
    
    for j=1:NPOSES
        % rotate into world
        v = wRb_cams_estimate(:,:,j) * u(:,j);
        B = eye(3,3) - v*v';
        A = A + B;
        b = b + B*p_cams_estimate(:,:,j);
    end
    
    % solve
    points_world_estimate(:,:,i) = A\b;
end
% plot estimated 3D points
figure(f3d);
scatter3(points_world_estimate(1,:), points_world_estimate(2,:), points_world_estimate(3,:), 'r');

% find best point 
point_deltas = points_world - squeeze(points_world_estimate);
point_deltas = sqrt(sum(point_deltas.^2,1));
[~,best_point_idx] = min(point_deltas);

% convert poses to SE3
cam_pose_estimates = zeros(4,4,NPOSES);
for j=1:NPOSES
    wRb = wRb_cams_estimate(:,:,j);
    p = p_cams_estimate(:,:,j);
    cam_pose_estimates(:,:,j) = [wRb' -wRb'*p; 0 0 0 1];
end

% run bundle adjustment
NUM_ITERATIONS = 10;

% we will optimize only the poses from START_POSE to NPOSES (inclusive)
START_POSE = 3;
NPOSES_OPT = (NPOSES - START_POSE + 1);

for iter=1:NUM_ITERATIONS
    % formulate jacobian and residual 
    J = zeros(NPTS*NPOSES*2, NPTS*3 + NPOSES_OPT*6);
    r = zeros(NPTS*NPOSES*2, 1);

    % structure of jacobian: 
    % [points0 .. pointsi .. pointsN | poses0 .. posesi .. posesN]
    for i=1:NPTS
        p_world = points_world_estimate(:,:,i);
        for j=1:NPOSES
            % camera pose
            H_cam = cam_pose_estimates(:,:,j);
            
            % transform to camera
            p_cam = H_cam * [p_world; 1];
            p_cam = p_cam(1:3); % truncate to remove 1
            
            xc = p_cam(1);  % camera coordinate
            yc = p_cam(2);
            zc = p_cam(3);
            
            % projection jacobian (2x3)
            Jproj = [1/zc 0 -xc/(zc*zc); 
                    0 1/zc -yc/(zc*zc)]; 
            
            % project to image coordinates and calculate residual
            h_est = p_cam / p_cam(3);      
            row = (j-1)*NPTS*2 + (i-1)*2 + 1;
            r(row:row+1,1) = points_image_noisy(1:2,i,j) - h_est(1:2);
            
            % pose jacobian (3x6)
            Jpose = [eye(3,3) -skew3(p_cam)];
            
            % point jacobian (3x3)
            Jpoint = H_cam(1:3,1:3);
            
            % insert jacobians
            if (j >= START_POSE)
                % optimizing pose also
                cols_pose = NPTS*3 + (j-START_POSE)*6 + 1;
                cols_pose = cols_pose:(cols_pose+5);
                J(row:row+1, cols_pose) = Jproj * Jpose;
            else
                % optimizing only point
            end
            
            cols_point = (i-1)*3 + 1;
            cols_point = cols_point:(cols_point+2);
            J(row:row+1, cols_point) = Jproj * Jpoint;
        end
    end
    
    fprintf('Iter %i, magnitude %f\n', iter, norm(r));
    
    % calculate cauchy weights
    r2 = r.*r;
    sigsqrd = mean(r2);
    W = 1 ./ (1 + r2/sigsqrd);
    W = diag(W);
    
    % calculate update (slow and simple method)
    H = J' * W * J;
    dx = H\(J' * W * r);
    
    % update points
    dx_points = dx(1:(NPTS*3),:);
    dx_points = reshape(dx_points,3,1,size(dx_points,1)/3);
    points_world_estimate = points_world_estimate + dx_points;
    
    % update poses
    dx_poses = dx((NPTS*3 + 1):end,:);
    dx_poses = reshape(dx_poses,6,size(dx_poses,1)/6);
    for j=START_POSE:NPOSES
        twist = dx_poses(:,j - START_POSE + 1);
        % approximate the exponential map
        S = skew3(twist(4:6));
        V = eye(3,3) + (1/2)*S + (1/6)*S*S;
        update = [rodrigues(twist(4:6)) V*twist(1:3); 0 0 0 1];
        cam_pose_estimates(:,:,j) = update * cam_pose_estimates(:,:,j);
    end
end

% convert poses back to R,p form
for j=1:NPOSES
    H = cam_pose_estimates(:,:,j);
    wRb = H(1:3,1:3)';
    p = -wRb * H(1:3,4);    
    wRb_cams_estimate(:,:,j) = wRb;
    p_cams_estimate(:,:,j) = p;
end

% plot adjusted points
figure(f3d);
scatter3(points_world_estimate(1,:), points_world_estimate(2,:), points_world_estimate(3,:), 'g');

% output positions of cameras after adjustment
disp(squeeze(p_cams));
disp(squeeze(p_cams_noisy));
disp(squeeze(p_cams_estimate));
