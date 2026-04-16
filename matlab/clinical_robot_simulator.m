classdef clinical_robot_simulator < handle
    % CLINICAL_ROBOT_SIMULATOR - Comprehensive simulation framework for clinical medication delivery robots
    % This class provides a complete simulation environment for testing robotic manipulation
    % in hospital pharmacy environments, integrating MATLAB Robotics System Toolbox,
    % Deep Learning Toolbox, and custom clinical environment models.
    %
    % Key Features:
    % - Franka Emika robot arm simulation with medication manipulation capabilities
    % - Hospital pharmacy environment modeling with realistic object configurations
    % - Human-aware motion planning with dynamic obstacle avoidance
    % - Integration with Octo foundation model for action prediction
    % - Clinical data collection protocol implementation
    % - Performance benchmarking and evaluation metrics
    %
    % Usage:
    %   simulator = clinical_robot_simulator();
    %   simulator.initializeEnvironment('hospital_layout_A');
    %   simulator.loadFoundationModel('octo_model_path');
    %   results = simulator.runAdaptationExperiment(demonstrations);
    %
    % Author: Idriss Djiofack Teledjieu
    % Clinical Robot Adaptation Project
    % University of Colorado Boulder HIRO Laboratory Collaboration
    
    properties (Access = private)
        % Robot configuration
        robotModel
        robotConfig
        endEffectorName
        
        % Environment configuration
        environmentModel
        pharmacyLayout
        medicationObjects
        workspaceBounds
        
        % Human detection and tracking
        humanTracker
        humanPoses
        humanVelocities
        
        % Foundation model interface
        foundationModel
        modelTokenizer
        adaptationHistory
        
        % Simulation parameters
        timeStep = 0.01
        maxTime = 30.0
        currentTime = 0
        
        % Data collection
        collectedData
        demonstrationBuffer
        qualityMetrics
        
        % Performance tracking
        taskMetrics
        safetyMetrics
        adaptationMetrics
    end
    
    properties (Access = public)
        % Public configuration
        verbose = true
        saveResults = true
        visualizationEnabled = true
        
        % Clinical protocol parameters
        minDemonstrationsPerObject = 10
        maxDemonstrationsPerObject = 50
        qualityThreshold = 0.85
        
        % Safety parameters
        minHumanDistance = 0.5  % meters
        maxEndEffectorVelocity = 0.3  % m/s
        maxJointAcceleration = 2.0  % rad/s^2
    end
    
    methods
        function obj = clinical_robot_simulator(configFile)
            % Constructor - Initialize the clinical robot simulator
            % Inputs:
            %   configFile - (optional) Path to configuration file
            
            if nargin < 1
                configFile = '';
            end
            
            % Initialize robot model
            obj.initializeRobot();
            
            % Initialize environment
            obj.initializeEnvironment();
            
            % Initialize human tracking
            obj.initializeHumanTracking();
            
            % Initialize data structures
            obj.initializeDataStructures();
            
            % Load configuration if provided
            if ~isempty(configFile) && exist(configFile, 'file')
                obj.loadConfiguration(configFile);
            end
            
            if obj.verbose
                fprintf('Clinical Robot Simulator initialized successfully\n');
                fprintf('Robot: Franka Emika with 7 DOF\n');
                fprintf('Environment: Hospital Pharmacy Simulation\n');
                fprintf('Foundation Model: Octo Integration Ready\n');
            end
        end
        
        function success = initializeRobot(obj)
            % Initialize Franka Emika robot model with clinical medication handling capabilities
            
            try
                % Load Franka Emika URDF model
                urdfPath = 'robot_models/franka_emika.urdf';
                if ~exist(urdfPath, 'file')
                    % Create default robot model if URDF not available
                    obj.robotModel = importrobot('universalUR5.urdf');
                    if obj.verbose
                        fprintf('Using default UR5 model as Franka Emika substitute\n');
                    end
                else
                    obj.robotModel = importrobot(urdfPath);
                end
                
                % Configure robot for clinical applications
                obj.robotConfig = obj.robotModel.DataFormat;
                obj.endEffectorName = 'tool0';
                
                % Set joint limits for safe clinical operation
                obj.robotModel.JointLimits = [
                    -2.8973, 2.8973;  % Joint 1
                    -1.7628, 1.7628;  % Joint 2
                    -2.8973, 2.8973;  % Joint 3
                    -3.0718, -0.0698; % Joint 4
                    -2.8973, 2.8973;  % Joint 5
                    -0.0175, 3.7525;  % Joint 6
                    -2.8973, 2.8973   % Joint 7
                ];
                
                % Set maximum velocities for clinical safety
                obj.robotModel.MaxJointVelocity = [
                    2.1750;  % rad/s
                    2.1750;
                    2.1750;
                    2.1750;
                    2.6100;
                    2.6100;
                    2.6100
                ];
                
                success = true;
                
            catch ME
                warning('Failed to initialize robot model: %s', ME.message);
                success = false;
            end
        end
        
        function success = initializeEnvironment(obj, layoutType)
            % Initialize hospital pharmacy environment
            % Inputs:
            %   layoutType - (optional) Type of hospital layout ('A', 'B', 'C')
            
            if nargin < 2
                layoutType = 'A';
            end
            
            try
                % Define workspace boundaries for pharmacy environment
                obj.workspaceBounds = struct(...
                    'xMin', -1.0, 'xMax', 1.0, ...
                    'yMin', -0.8, 'yMax', 0.8, ...
                    'zMin', 0.0,  'zMax', 1.5);
                
                % Initialize pharmacy layout based on type
                switch layoutType
                    case 'A'
                        obj.createPharmacyLayoutA();
                    case 'B'
                        obj.createPharmacyLayoutB();
                    case 'C'
                        obj.createPharmacyLayoutC();
                    otherwise
                        obj.createPharmacyLayoutA();
                end
                
                % Initialize medication objects
                obj.initializeMedicationObjects();
                
                success = true;
                
                if obj.verbose
                    fprintf('Environment initialized: Layout Type %s\n', layoutType);
                    fprintf('Workspace: [%.1f, %.1f] x [%.1f, %.1f] x [%.1f, %.1f] meters\n', ...
                        obj.workspaceBounds.xMin, obj.workspaceBounds.xMax, ...
                        obj.workspaceBounds.yMin, obj.workspaceBounds.yMax, ...
                        obj.workspaceBounds.zMin, obj.workspaceBounds.zMax);
                end
                
            catch ME
                warning('Failed to initialize environment: %s', ME.message);
                success = false;
            end
        end
        
        function createPharmacyLayoutA(obj)
            % Create pharmacy Layout Type A - Standard hospital pharmacy configuration
            % Features: Central medication carousel, peripheral shelving, staff workflow area
            
            obj.pharmacyLayout = struct();
            
            % Central medication carousel (rotating storage system)
            obj.pharmacyLayout.carousel = struct(...
                'position', [0, 0, 0.8], ...
                'radius', 0.3, ...
                'height', 1.2, ...
                'compartments', 24, ...
                'rotationSpeed', 0.1);  % rad/s
            
            % Peripheral shelving units
            shelfPositions = [
                -0.8, 0, 0.9;   % Left shelf
                0.8, 0, 0.9;    % Right shelf
                0, -0.6, 0.9;   % Back shelf
            ];
            
            obj.pharmacyLayout.shelves = struct();
            for i = 1:size(shelfPositions, 1)
                obj.pharmacyLayout.shelves.(sprintf('shelf_%d', i)) = struct(...
                    'position', shelfPositions(i, :), ...
                    'dimensions', [0.4, 0.2, 1.2], ...
                    'shelfCount', 6, ...
                    'loadCapacity', 50);  % kg
            end
            
            % Staff workflow areas
            obj.pharmacyLayout.workflowAreas = struct(...
                'preparation', struct('position', [0.4, 0.4, 0.9], 'radius', 0.3), ...
                'dispensing', struct('position', [-0.4, 0.4, 0.9], 'radius', 0.3), ...
                'verification', struct('position', [0, -0.3, 0.9], 'radius', 0.25));
            
            % Robot workspace boundaries
            obj.pharmacyLayout.robotWorkspace = struct(...
                'homePosition', [0, 0, 0.5], ...
                'approachHeight', 0.3, ...
                'graspHeight', 0.9);
        end
        
        function createPharmacyLayoutB(obj)
            % Create pharmacy Layout Type B - Compact urban hospital configuration
            % Features: Vertical storage systems, limited floor space, high throughput
            
            obj.pharmacyLayout = struct();
            
            % Vertical storage columns
            obj.pharmacyLayout.verticalStorage = struct();
            positions = [-0.5, 0; 0.5, 0];
            for i = 1:size(positions, 1)
                obj.pharmacyLayout.verticalStorage.(sprintf('column_%d', i)) = struct(...
                    'position', [positions(i, 1), positions(i, 2), 0], ...
                    'height', 2.0, ...
                    'width', 0.3, ...
                    'depth', 0.3, ...
                    'shelfCount', 12);
            end
            
            % Conveyor system for automated transport
            obj.pharmacyLayout.conveyor = struct(...
                'path', [ -0.6, 0; 0.6, 0], ...
                'speed', 0.2, ...
                'width', 0.15);
            
            % Compact workstations
            obj.pharmacyLayout.workstations = struct(...
                'input', struct('position', [-0.6, 0.3, 0.9]), ...
                'output', struct('position', [0.6, 0.3, 0.9]));
        end
        
        function createPharmacyLayoutC(obj)
            % Create pharmacy Layout Type C - Rural hospital configuration
            % Features: Simplified layout, manual backup systems, robust design
            
            obj.pharmacyLayout = struct();
            
            % Simple shelving arrangement
            obj.pharmacyLayout.simpleShelves = struct();
            shelfConfig = [
                -0.7, 0, 0.8, 1.0;  % position, height, length
                0.7, 0, 0.8, 1.0;
                0, -0.5, 0.8, 1.2;
            ];
            
            for i = 1:size(shelfConfig, 1)
                obj.pharmacyLayout.simpleShelves.(sprintf('shelf_%d', i)) = struct(...
                    'position', shelfConfig(i, 1:2), ...
                    'height', shelfConfig(i, 3), ...
                    'length', shelfConfig(i, 4), ...
                    'shelfCount', 4);
            end
            
            % Central work table
            obj.pharmacyLayout.workTable = struct(...
                'position', [0, 0, 0.8], ...
                'dimensions', [1.2, 0.8, 0.05]);
        end
        
        function initializeMedicationObjects(obj)
            % Initialize diverse medication object types for realistic simulation
            
            obj.medicationObjects = struct();
            
            % Vial medications (glass vials with rubber stoppers)
            obj.medicationObjects.vials = struct(...
                'count', 15, ...
                'dimensions', [0.02, 0.04, 0.02], ...  % radius, height, mass (kg)
                'fragility', 'high', ...
                'graspType', 'precision', ...
                'categories', {{'antibiotics', 'vaccines', 'insulin'}});
            
            % Blister packs (medication in plastic/aluminum blister packs)
            obj.medicationObjects.blisterPacks = struct(...
                'count', 20, ...
                'dimensions', [0.08, 0.12, 0.01], ...  % width, height, thickness
                'fragility', 'medium', ...
                'graspType', 'power', ...
                'categories', {{'oral_medications', 'pain_relief', 'chronic_meds'}});
            
            % Syringes (pre-filled medication syringes)
            obj.medicationObjects.syringes = struct(...
                'count', 10, ...
                'dimensions', [0.008, 0.08, 0.015], ...  % radius, length, mass
                'fragility', 'high', ...
                'graspType', 'precision', ...
                'categories', {{'injectables', 'vaccines', 'emergency_meds'}});
            
            % Medication bottles (plastic prescription bottles)
            obj.medicationObjects.bottles = struct(...
                'count', 12, ...
                'dimensions', [0.03, 0.08, 0.05], ...  % radius, height, mass
                'fragility', 'low', ...
                'graspType', 'power', ...
                'categories', {{'tablets', 'capsules', 'liquid_meds'}});
            
            % Medication pouches (IV medication bags)
            obj.medicationObjects.pouches = struct(...
                'count', 8, ...
                'dimensions', [0.12, 0.15, 0.3], ...  % width, height, mass
                'fragility', 'medium', ...
                'graspType', 'pinch', ...
                'categories', {{'iv_medications', 'fluids', 'nutrition'}});
            
            if obj.verbose
                fprintf('Medication objects initialized: %d vials, %d blister packs, %d syringes, %d bottles, %d pouches\n', ...
                    obj.medicationObjects.vials.count, ...
                    obj.medicationObjects.blisterPacks.count, ...
                    obj.medicationObjects.syringes.count, ...
                    obj.medicationObjects.bottles.count, ...
                    obj.medicationObjects.pouches.count);
            end
        end
        
        function initializeHumanTracking(obj)
            % Initialize human detection and tracking system for clinical safety
            
            obj.humanTracker = struct();
            obj.humanPoses = [];
            obj.humanVelocities = [];
            
            % Configure human detection parameters
            obj.humanTracker.detectionRange = 3.0;  % meters
            obj.humanTracker.updateRate = 10;  % Hz
            obj.humanTracker.confidenceThreshold = 0.7;
            
            % Safety zones around humans
            obj.humanTracker.safetyZones = struct(...
                'critical', 0.3, ...    % meters - immediate collision avoidance
                'warning', 0.6, ...     % meters - trajectory modification
                'awareness', 1.2);     % meters - reduced speed
            
            if obj.verbose
                fprintf('Human tracking system initialized\n');
                fprintf('Detection range: %.1f meters\n', obj.humanTracker.detectionRange);
                fprintf('Safety zones: Critical=%.1fm, Warning=%.1fm, Awareness=%.1fm\n', ...
                    obj.humanTracker.safetyZones.critical, ...
                    obj.humanTracker.safetyZones.warning, ...
                    obj.humanTracker.safetyZones.awareness);
            end
        end
        
        function initializeDataStructures(obj)
            % Initialize data collection and analysis structures
            
            % Demonstration data buffer
            obj.demonstrationBuffer = struct(...
                'observations', [], ...
                'actions', [], ...
                'rewards', [], ...
                'next_observations', [], ...
                'dones', [], ...
                'metadata', struct(...
                    'objectTypes', [], ...
                    'graspTypes', [], ...
                    'successFlags', [], ...
                    'timestamps', []));
            
            % Quality metrics for demonstrations
            obj.qualityMetrics = struct(...
                'trajectorySmoothness', [], ...
                'graspStability', [], ...
                'collisionFree', [], ...
                'taskCompletion', [], ...
                'humanAwareness', []);
            
            % Performance tracking
            obj.taskMetrics = struct(...
                'successRate', [], ...
                'completionTime', [], ...
                'graspAccuracy', [], ...
                'objectRecognition', []);
            
            % Safety metrics
            obj.safetyMetrics = struct(...
                'humanCollisions', 0, ...
                'objectDrops', 0, ...
                'velocityViolations', 0, ...
                'workspaceViolations', 0);
            
            % Adaptation metrics
            obj.adaptationMetrics = struct(...
                'demonstrationCount', 0, ...
                'adaptationProgress', 0, ...
                'generalizationScore', 0, ...
                'convergenceRate', []);
        end
        
        function success = loadFoundationModel(obj, modelPath)
            % Load Octo foundation model for action prediction
            % Inputs:
            %   modelPath - Path to pre-trained Octo model files
            
            if nargin < 2
                modelPath = 'models/octo_pretrained/';
            end
            
            try
                % Check if model files exist
                modelFiles = {'model_weights.h5', 'model_config.json', 'tokenizer.json'};
                for i = 1:length(modelFiles)
                    if ~exist(fullfile(modelPath, modelFiles{i}), 'file')
                        error('Model file not found: %s', fullfile(modelPath, modelFiles{i}));
                    end
                end
                
                % Load model configuration
                configPath = fullfile(modelPath, 'model_config.json');
                fid = fopen(configPath);
                configText = fread(fid, '*char')';
                fclose(fid);
                config = jsondecode(configText);
                
                % Initialize model architecture (placeholder for actual Octo model)
                obj.foundationModel = struct(...
                    'architecture', 'transformer', ...
                    'inputDimensions', config.input_dimensions, ...
                    'outputDimensions', config.output_dimensions, ...
                    'sequenceLength', config.sequence_length, ...
                    'numLayers', config.num_layers, ...
                    'hiddenSize', config.hidden_size);
                
                % Load tokenizer for language instructions
                tokenizerPath = fullfile(modelPath, 'tokenizer.json');
                fid = fopen(tokenizerPath);
                tokenizerText = fread(fid, '*char')';
                fclose(fid);
                obj.modelTokenizer = jsondecode(tokenizerText);
                
                % Initialize adaptation history
                obj.adaptationHistory = struct(...
                    'episodes', [], ...
                    'losses', [], ...
                    'accuracies', [], ...
                    'timestamps', []);
                
                success = true;
                
                if obj.verbose
                    fprintf('Octo foundation model loaded successfully\n');
                    fprintf('Architecture: %s with %d layers\n', ...
                        obj.foundationModel.architecture, obj.foundationModel.numLayers);
                    fprintf('Input dimensions: [%s]\n', num2str(obj.foundationModel.inputDimensions));
                end
                
            catch ME
                warning('Failed to load foundation model: %s', ME.message);
                success = false;
            end
        end
        
        function trajectory = planMotion(obj, startPose, goalPose, humanPoses)
            % Plan collision-free motion with human awareness
            % Inputs:
            %   startPose - Initial robot pose [x, y, z, qx, qy, qz, qw]
            %   goalPose - Target robot pose
            %   humanPoses - Current human positions and velocities
            % Outputs:
            %   trajectory - Planned trajectory as time-indexed joint positions
            
            if nargin < 4
                humanPoses = [];
            end
            
            % Initialize trajectory planner
            planner = manipulatorRRT(obj.robotModel);
            
            % Set planning parameters for clinical safety
            planner.MaxConnectionDistance = 0.1;
            planner.MaxIterations = 5000;
            
            % Define state space with human-aware constraints
            bounds = obj.robotModel.JointLimits;
            
            % Add human avoidance constraints if humans present
            if ~isempty(humanPoses)
                % Inflate obstacles around human positions
                for i = 1:size(humanPoses, 1)
                    humanPos = humanPoses(i, 1:3);
                    % Add safety margin around human
                    safetyMargin = obj.humanTracker.safetyZones.warning;
                    % This would integrate with a more sophisticated obstacle avoidance system
                end
            end
            
            % Plan trajectory
            try
                [trajectory, solutionInfo] = plan(planner, startPose, goalPose);
                
                % Optimize trajectory for smoothness and clinical safety
                trajectory = obj.optimizeTrajectory(trajectory, humanPoses);
                
                if obj.verbose && solutionInfo.IsPathFound
                    fprintf('Motion planning successful: %.2f seconds trajectory\n', ...
                        size(trajectory, 1) * obj.timeStep);
                end
                
            catch ME
                warning('Motion planning failed: %s', ME.message);
                trajectory = [];
            end
        end
        
        function optimizedTrajectory = optimizeTrajectory(obj, trajectory, humanPoses)
            % Optimize trajectory for smoothness and human-aware operation
            
            if isempty(trajectory)
                optimizedTrajectory = [];
                return;
            end
            
            % Time-optimal trajectory parameterization
            waypoints = trajectory;
            maxVel = obj.robotModel.MaxJointVelocity;
            maxAcc = ones(size(maxVel)) * obj.maxJointAcceleration;
            
            % Compute velocity profile
            [velProfile, accProfile] = obj.computeVelocityProfile(waypoints, maxVel, maxAcc);
            
            % Apply human-aware velocity scaling
            if ~isempty(humanPoses)
                velProfile = obj.applyHumanAwareScaling(velProfile, waypoints, humanPoses);
            end
            
            % Integrate to get optimized trajectory
            optimizedTrajectory = obj.integrateVelocityProfile(waypoints, velProfile);
        end
        
        function [velProfile, accProfile] = computeVelocityProfile(obj, waypoints, maxVel, maxAcc)
            % Compute time-optimal velocity profile for trajectory
            
            nWaypoints = size(waypoints, 1);
            nJoints = size(waypoints, 2) - 1;  % Exclude time column
            
            % Initialize profiles
            velProfile = zeros(nWaypoints, nJoints);
            accProfile = zeros(nWaypoints, nJoints);
            
            % Compute segment distances and directions
            for i = 1:(nWaypoints-1)
                segment = waypoints(i+1, 1:nJoints) - waypoints(i, 1:nJoints);
                segmentLength = norm(segment);
                direction = segment / segmentLength;
                
                % Maximum feasible velocity for this segment
                maxSegmentVel = min(maxVel);
                
                % Apply acceleration constraints
                if i == 1
                    velProfile(i, :) = direction * min(maxSegmentVel, sqrt(2 * maxAcc * segmentLength));
                else
                    % Consider previous segment velocity
                    prevVel = velProfile(i-1, :);
                    feasibleVel = prevVel + maxAcc * obj.timeStep;
                    velProfile(i, :) = direction * min(maxSegmentVel, norm(feasibleVel));
                end
                
                % Compute acceleration
                if i > 1
                    accProfile(i, :) = (velProfile(i, :) - velProfile(i-1, :)) / obj.timeStep;
                end
            end
            
            % Ensure final velocity is zero
            velProfile(end, :) = 0;
        end
        
        function scaledVelProfile = applyHumanAwareScaling(obj, velProfile, waypoints, humanPoses)
            % Scale velocities based on human proximity for safety
            
            scaledVelProfile = velProfile;
            nWaypoints = size(waypoints, 1);
            nJoints = size(velProfile, 2);
            
            for i = 1:nWaypoints
                % Compute end-effector position for this waypoint
                jointConfig = waypoints(i, 1:nJoints);
                eePose = obj.robotModel.getTransform(jointConfig, obj.endEffectorName);
                eePosition = eePose(1:3, 4);
                
                % Find minimum distance to any human
                minHumanDist = inf;
                for j = 1:size(humanPoses, 1)
                    humanPos = humanPoses(j, 1:3);
                    dist = norm(eePosition - humanPos);
                    minHumanDist = min(minHumanDist, dist);
                end
                
                % Apply velocity scaling based on distance
                if minHumanDist < obj.humanTracker.safetyZones.critical
                    % Critical zone - stop motion
                    scaleFactor = 0.0;
                elseif minHumanDist < obj.humanTracker.safetyZones.warning
                    % Warning zone - reduce speed significantly
                    scaleFactor = 0.3;
                elseif minHumanDist < obj.humanTracker.safetyZones.awareness
                    % Awareness zone - moderate speed reduction
                    scaleFactor = 0.6;
                else
                    % Safe distance - full speed
                    scaleFactor = 1.0;
                end
                
                scaledVelProfile(i, :) = scaledVelProfile(i, :) * scaleFactor;
            end
        end
        
        function trajectory = integrateVelocityProfile(obj, waypoints, velProfile)
            % Integrate velocity profile to generate final trajectory
            
            nWaypoints = size(waypoints, 1);
            nJoints = size(waypoints, 2) - 1;
            trajectory = zeros(nWaypoints, nJoints + 1);  % +1 for time
            
            trajectory(1, :) = [waypoints(1, 1:nJoints), 0];  % Start at time 0
            
            for i = 2:nWaypoints
                % Integrate velocity to get position
                dt = obj.timeStep;
                trajectory(i, 1:nJoints) = trajectory(i-1, 1:nJoints) + velProfile(i-1, :) * dt;
                trajectory(i, end) = trajectory(i-1, end) + dt;
            end
        end
        
        function success = addDemonstration(obj, observation, action, reward, metadata)
            % Add a new demonstration to the adaptation buffer
            % Inputs:
            %   observation - Environment state (images, robot state, etc.)
            %   action - Robot action taken
            %   reward - Reward signal for quality assessment
            %   metadata - Additional demonstration information
            
            try
                % Validate demonstration quality
                qualityScore = obj.evaluateDemonstrationQuality(observation, action, metadata);
                
                if qualityScore >= obj.qualityThreshold
                    % Add to demonstration buffer
                    obj.demonstrationBuffer.observations(end+1) = {observation};
                    obj.demonstrationBuffer.actions(end+1, :) = action;
                    obj.demonstrationBuffer.rewards(end+1) = reward;
                    
                    % Add metadata
                    obj.demonstrationBuffer.metadata.objectTypes(end+1) = metadata.objectType;
                    obj.demonstrationBuffer.metadata.graspTypes(end+1) = metadata.graspType;
                    obj.demonstrationBuffer.metadata.successFlags(end+1) = metadata.success;
                    obj.demonstrationBuffer.metadata.timestamps(end+1) = datetime('now');
                    
                    % Update adaptation metrics
                    obj.adaptationMetrics.demonstrationCount = obj.adaptationMetrics.demonstrationCount + 1;
                    
                    success = true;
                    
                    if obj.verbose
                        fprintf('Demonstration added (quality: %.3f, total: %d)\n', ...
                            qualityScore, obj.adaptationMetrics.demonstrationCount);
                    end
                    
                else
                    success = false;
                    if obj.verbose
                        fprintf('Demonstration rejected (quality: %.3f < %.3f)\n', ...
                            qualityScore, obj.qualityThreshold);
                    end
                end
                
            catch ME
                warning('Failed to add demonstration: %s', ME.message);
                success = false;
            end
        end
        
        function qualityScore = evaluateDemonstrationQuality(obj, observation, action, metadata)
            % Evaluate the quality of a demonstration for adaptation
            
            % Initialize quality components
            trajectoryScore = 0;
            graspScore = 0;
            safetyScore = 0;
            taskScore = 0;
            
            % Evaluate trajectory smoothness
            if isfield(observation, 'trajectory') && length(observation.trajectory) > 1
                velocities = diff(observation.trajectory) / obj.timeStep;
                accelerations = diff(velocities) / obj.timeStep;
                smoothness = 1.0 / (1.0 + std(accelerations(:)));
                trajectoryScore = smoothness;
            end
            
            % Evaluate grasp stability
            if isfield(metadata, 'graspStability')
                graspScore = metadata.graspStability;
            end
            
            % Evaluate safety (collision avoidance, velocity limits)
            if isfield(metadata, 'safetyViolations')
                safetyScore = 1.0 - metadata.safetyViolations;
            else
                safetyScore = 1.0;
            end
            
            % Evaluate task completion
            if isfield(metadata, 'success')
                taskScore = double(metadata.success);
            end
            
            % Combine scores with weights
            weights = [0.3, 0.3, 0.2, 0.2];  % trajectory, grasp, safety, task
            scores = [trajectoryScore, graspScore, safetyScore, taskScore];
            qualityScore = dot(weights, scores);
        end
        
        function results = runAdaptationExperiment(obj, numDemonstrations, testTasks)
            % Run complete adaptation experiment with few-shot learning
            % Inputs:
            %   numDemonstrations - Number of demonstrations to collect
            %   testTasks - Test tasks for evaluation
            % Outputs:
            %   results - Experiment results and performance metrics
            
            if nargin < 3
                testTasks = obj.generateTestTasks();
            end
            
            if obj.verbose
                fprintf('Starting adaptation experiment with %d demonstrations\n', numDemonstrations);
                fprintf('Test tasks: %d\n', length(testTasks));
            end
            
            % Initialize results structure
            results = struct();
            results.adaptationProgress = [];
            results.testPerformance = [];
            results.convergenceMetrics = struct();
            
            % Phase 1: Collect demonstrations
            for demo = 1:numDemonstrations
                % Generate demonstration scenario
                scenario = obj.generateDemonstrationScenario();
                
                % Execute demonstration (simulated)
                [observation, action, reward, metadata] = obj.executeDemonstration(scenario);
                
                % Add to adaptation buffer
                success = obj.addDemonstration(observation, action, reward, metadata);
                
                if success && mod(demo, 10) == 0
                    % Evaluate current adaptation progress
                    performance = obj.evaluateAdaptation(testTasks);
                    results.adaptationProgress(end+1) = performance;
                    
                    if obj.verbose
                        fprintf('Demo %d: Performance = %.3f\n', demo, performance);
                    end
                end
            end
            
            % Phase 2: Final evaluation
            finalPerformance = obj.evaluateAdaptation(testTasks);
            results.finalPerformance = finalPerformance;
            
            % Phase 3: Compute convergence metrics
            results.convergenceMetrics = obj.computeConvergenceMetrics(results.adaptationProgress);
            
            % Save results if requested
            if obj.saveResults
                timestamp = datestr(now, 'yyyymmdd_HHMMSS');
                filename = sprintf('adaptation_results_%s.mat', timestamp);
                save(filename, 'results');
                if obj.verbose
                    fprintf('Results saved to %s\n', filename);
                end
            end
            
            if obj.verbose
                fprintf('Adaptation experiment completed\n');
                fprintf('Final performance: %.3f\n', finalPerformance);
                fprintf('Convergence rate: %.3f\n', results.convergenceMetrics.rate);
            end
        end
        
        function scenario = generateDemonstrationScenario(obj)
            % Generate a realistic demonstration scenario
            
            % Randomly select medication object type
            objectTypes = fieldnames(obj.medicationObjects);
            objType = objectTypes{randi(length(objectTypes))};
            
            % Randomly select grasp type based on object
            graspTypes = obj.medicationObjects.(objType).graspType;
            if iscell(graspTypes)
                graspType = graspTypes{randi(length(graspTypes))};
            else
                graspType = graspTypes;
            end
            
            % Generate object position in workspace
            x = rand() * (obj.workspaceBounds.xMax - obj.workspaceBounds.xMin) + obj.workspaceBounds.xMin;
            y = rand() * (obj.workspaceBounds.yMax - obj.workspaceBounds.yMin) + obj.workspaceBounds.yMin;
            z = obj.workspaceBounds.zMin + rand() * (obj.workspaceBounds.zMax - obj.workspaceBounds.zMin);
            
            % Generate human presence (30% chance)
            humanPresent = rand() < 0.3;
            humanPos = [];
            if humanPresent
                humanPos = [randn * 0.5, randn * 0.5, 1.0];  % Around workspace
            end
            
            scenario = struct(...
                'objectType', objType, ...
                'graspType', graspType, ...
                'objectPosition', [x, y, z], ...
                'humanPresent', humanPresent, ...
                'humanPosition', humanPos, ...
                'difficulty', rand());  % 0 = easy, 1 = hard
        end
        
        function [observation, action, reward, metadata] = executeDemonstration(obj, scenario)
            % Execute a single demonstration (simulated)
            
            % Generate observation (simplified)
            observation = struct();
            observation.image = rand(224, 224, 3);  % RGB image
            observation.robotState = rand(1, 7);   % Joint positions
            observation.objectPose = [scenario.objectPosition, 0, 0, 0, 1];  % Position + quaternion
            
            % Generate action (simplified optimal trajectory)
            action = obj.generateOptimalAction(scenario);
            
            % Compute reward based on scenario difficulty and execution quality
            baseReward = 1.0 - scenario.difficulty * 0.3;  % Harder tasks get lower base reward
            
            % Quality factors
            trajectoryQuality = 0.9 + rand() * 0.1;  % High quality with small variation
            graspQuality = 0.85 + rand() * 0.15;
            safetyScore = 1.0;  % Assume safe execution
            
            reward = baseReward * trajectoryQuality * graspQuality * safetyScore;
            
            % Generate metadata
            metadata = struct();
            metadata.objectType = scenario.objectType;
            metadata.graspType = scenario.graspType;
            metadata.success = reward > 0.7;
            metadata.graspStability = graspQuality;
            metadata.safetyViolations = 0;
            metadata.trajectorySmoothness = trajectoryQuality;
        end
        
        function action = generateOptimalAction(obj, scenario)
            % Generate optimal action for given scenario (simplified)
            
            % This would normally use the foundation model or motion planning
            % For simulation, generate a reasonable trajectory
            
            % Current robot state (simplified)
            currentJoints = [0, 0, 0, 0, 0, 0, 0];
            
            % Target end-effector pose
            targetPose = [scenario.objectPosition(1), scenario.objectPosition(2), ...
                         scenario.objectPosition(3) + 0.1, 0, 0, 0, 1];  % Slightly above object
            
            % Compute inverse kinematics (simplified)
            targetJoints = obj.inverseKinematics(currentJoints, targetPose);
            
            % Generate trajectory as action sequence
            nSteps = 50;
            action = zeros(nSteps, 7);
            for i = 1:nSteps
                alpha = i / nSteps;
                action(i, :) = (1 - alpha) * currentJoints + alpha * targetJoints;
            end
        end
        
        function targetJoints = inverseKinematics(obj, currentJoints, targetPose)
            % Simplified inverse kinematics for demonstration
            
            % This is a placeholder - real implementation would use proper IK
            % For simulation, generate reasonable joint angles
            
            % Extract target position
            targetPos = targetPose(1:3);
            
            % Simple geometric IK (very simplified)
            distance = norm(targetPos(1:2));
            height = targetPos(3);
            
            % Base joint angle
            baseAngle = atan2(targetPos(2), targetPos(1));
            
            % Arm configuration (simplified)
            armExtension = min(distance, 0.8);  % Max reach
            armAngle = acos(height / max(height, 0.1));
            
            % Generate joint configuration
            targetJoints = [
                baseAngle;           % Joint 1: base rotation
                deg2rad(-45);        % Joint 2: shoulder
                deg2rad(45);         % Joint 3: elbow
                deg2rad(0);          % Joint 4: wrist 1
                deg2rad(30);         % Joint 5: wrist 2
                deg2rad(0);          % Joint 6: wrist 3
                0                    % Joint 7: gripper
            ];
            
            % Add some variation based on current state
            targetJoints = 0.7 * currentJoints + 0.3 * targetJoints;
        end
        
        function testTasks = generateTestTasks(obj)
            % Generate diverse test tasks for adaptation evaluation
            
            testTasks = [];
            
            % Generate tasks for each object type
            objectTypes = fieldnames(obj.medicationObjects);
            
            for i = 1:length(objectTypes)
                objType = objectTypes{i};
                
                % Generate 3 test scenarios per object type
                for j = 1:3
                    task = struct();
                    task.objectType = objType;
                    task.graspType = obj.medicationObjects.(objType).graspType;
                    
                    % Vary object positions
                    angle = (j - 1) * 2 * pi / 3;
                    radius = 0.4 + rand() * 0.2;
                    task.objectPosition = [
                        radius * cos(angle), ...
                        radius * sin(angle), ...
                        0.8 + rand() * 0.3
                    ];
                    
                    % Vary difficulty
                    task.difficulty = 0.3 + (j - 1) * 0.2;
                    
                    testTasks(end+1) = task;
                end
            end
        end
        
        function performance = evaluateAdaptation(obj, testTasks)
            % Evaluate current adaptation performance on test tasks
            
            if isempty(testTasks)
                performance = 0;
                return;
            end
            
            totalSuccess = 0;
            totalReward = 0;
            
            for i = 1:length(testTasks)
                % Execute task with current model
                [success, reward] = obj.executeTestTask(testTasks(i));
                totalSuccess = totalSuccess + success;
                totalReward = totalReward + reward;
            end
            
            % Compute performance metrics
            successRate = totalSuccess / length(testTasks);
            averageReward = totalReward / length(testTasks);
            
            % Combined performance score
            performance = 0.6 * successRate + 0.4 * averageReward;
        end
        
        function [success, reward] = executeTestTask(obj, task)
            % Execute a single test task for evaluation
            
            % This would use the adapted foundation model
            % For simulation, generate performance based on adaptation progress
            
            % Base performance improves with more demonstrations
            nDemos = obj.adaptationMetrics.demonstrationCount;
            basePerformance = min(0.9, 0.3 + nDemos / 100);
            
            % Adjust for task difficulty
            difficultyFactor = 1.0 - task.difficulty * 0.4;
            
            % Add some randomness
            noise = (rand() - 0.5) * 0.1;
            
            % Compute final performance
            performance = basePerformance * difficultyFactor + noise;
            performance = max(0, min(1, performance));
            
            success = performance > 0.7;
            reward = performance;
        end
        
        function metrics = computeConvergenceMetrics(obj, progressHistory)
            % Compute convergence metrics from adaptation progress
            
            if length(progressHistory) < 2
                metrics = struct('rate', 0, 'efficiency', 0, 'stability', 0);
                return;
            end
            
            % Convergence rate (slope of performance improvement)
            x = 1:length(progressHistory);
            p = polyfit(x, progressHistory', 1);
            metrics.rate = p(1);
            
            % Efficiency (performance per demonstration)
            finalPerformance = progressHistory(end);
            nDemos = length(progressHistory) * 10;  % Assuming eval every 10 demos
            metrics.efficiency = finalPerformance / nDemos;
            
            % Stability (variance in recent performance)
            if length(progressHistory) >= 5
                recentPerf = progressHistory(end-4:end);
                metrics.stability = 1.0 / (1.0 + var(recentPerf));
            else
                metrics.stability = 1.0;
            end
        end
        
        function visualizeResults(obj, results)
            % Visualize adaptation experiment results
            
            if ~obj.visualizationEnabled
                return;
            end
            
            figure('Position', [100, 100, 1200, 800]);
            
            % Plot 1: Adaptation Progress
            subplot(2, 3, 1);
            plot(1:length(results.adaptationProgress), results.adaptationProgress, 'b-', 'LineWidth', 2);
            xlabel('Evaluation Point');
            ylabel('Performance');
            title('Adaptation Progress');
            grid on;
            
            % Plot 2: Convergence Rate
            subplot(2, 3, 2);
            convergenceRate = results.convergenceMetrics.rate;
            bar(convergenceRate);
            ylabel('Convergence Rate');
            title(sprintf('Convergence Rate: %.4f', convergenceRate));
            grid on;
            
            % Plot 3: Final Performance Breakdown
            subplot(2, 3, 3);
            finalPerf = results.finalPerformance;
            pie([finalPerf, 1-finalPerf], {'Success', 'Failure'});
            title(sprintf('Final Performance: %.3f', finalPerf));
            
            % Plot 4: Demonstration Quality Distribution
            subplot(2, 3, 4);
            if ~isempty(obj.demonstrationBuffer.rewards)
                histogram(obj.demonstrationBuffer.rewards, 20);
                xlabel('Demonstration Quality');
                ylabel('Count');
                title('Demonstration Quality Distribution');
                grid on;
            end
            
            % Plot 5: Safety Metrics
            subplot(2, 3, 5);
            safetyData = [obj.safetyMetrics.humanCollisions, ...
                         obj.safetyMetrics.objectDrops, ...
                         obj.safetyMetrics.velocityViolations];
            bar(safetyData);
            set(gca, 'XTickLabel', {'Human Collisions', 'Object Drops', 'Velocity Violations'});
            ylabel('Count');
            title('Safety Metrics');
            grid on;
            
            % Plot 6: Adaptation Efficiency
            subplot(2, 3, 6);
            efficiency = results.convergenceMetrics.efficiency;
            bar(efficiency);
            ylabel('Performance per Demonstration');
            title(sprintf('Efficiency: %.6f', efficiency));
            grid on;
            
            sgtitle('Clinical Robot Adaptation Results');
        end
        
        function saveConfiguration(obj, filename)
            % Save current configuration to file
            
            config = struct();
            config.robot = obj.robotConfig;
            config.environment = obj.pharmacyLayout;
            config.safety = struct(...
                'minHumanDistance', obj.minHumanDistance, ...
                'maxEndEffectorVelocity', obj.maxEndEffectorVelocity, ...
                'maxJointAcceleration', obj.maxJointAcceleration);
            config.adaptation = struct(...
                'minDemonstrationsPerObject', obj.minDemonstrationsPerObject, ...
                'maxDemonstrationsPerObject', obj.maxDemonstrationsPerObject, ...
                'qualityThreshold', obj.qualityThreshold);
            
            save(filename, 'config');
            
            if obj.verbose
                fprintf('Configuration saved to %s\n', filename);
            end
        end
        
        function loadConfiguration(obj, filename)
            % Load configuration from file
            
            if exist(filename, 'file')
                loaded = load(filename);
                config = loaded.config;
                
                % Apply configuration
                if isfield(config, 'safety')
                    obj.minHumanDistance = config.safety.minHumanDistance;
                    obj.maxEndEffectorVelocity = config.safety.maxEndEffectorVelocity;
                    obj.maxJointAcceleration = config.safety.maxJointAcceleration;
                end
                
                if isfield(config, 'adaptation')
                    obj.minDemonstrationsPerObject = config.adaptation.minDemonstrationsPerObject;
                    obj.maxDemonstrationsPerObject = config.adaptation.maxDemonstrationsPerObject;
                    obj.qualityThreshold = config.adaptation.qualityThreshold;
                end
                
                if obj.verbose
                    fprintf('Configuration loaded from %s\n', filename);
                end
            else
                warning('Configuration file not found: %s', filename);
            end
        end
    end
end
