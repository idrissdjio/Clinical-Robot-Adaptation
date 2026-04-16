classdef human_aware_controller < handle
    % HUMAN_AWARE_CONTROLLER - Advanced human-aware motion control for clinical robots
    % This class implements sophisticated human detection, tracking, and motion planning
    % for safe robot operation in dynamic clinical environments. It integrates multiple
    % sensing modalities and predictive models to ensure safe human-robot collaboration.
    %
    % Key Features:
    % - Multi-modal human detection (vision, depth, thermal)
    % - Intent prediction using LSTM networks
    % - Dynamic safety zones with adaptive scaling
    % - Real-time trajectory optimization with social constraints
    % - Emergency stop and recovery protocols
    % - Clinical workflow integration
    %
    % Author: Idriss Djiofack Teledjieu
    % Clinical Robot Adaptation Project
    % HIRO Laboratory, University of Colorado Boulder
    
    properties (Access = private)
        % Human detection and tracking
        humanDetector
        poseEstimator
        intentPredictor
        trackingFilter
        
        % Safety zone management
        safetyZones
        zoneHistory
        violationHistory
        
        % Motion planning
        trajectoryOptimizer
        constraintGenerator
        emergencyPlanner
        
        % Sensor fusion
        visionProcessor
        depthProcessor
        thermalProcessor
        fusionFilter
        
        % Clinical context
        workflowAnalyzer
        activityRecognizer
        safetyProtocols
        
        % Performance monitoring
        safetyMetrics
        efficiencyMetrics
        interactionMetrics
    end
    
    properties (Access = public)
        % Configuration parameters
        detectionRange = 3.0          % meters
        predictionHorizon = 2.0       % seconds
        minSafetyDistance = 0.5      % meters
        emergencyThreshold = 0.3      % meters
        
        % Control parameters
        maxVelocity = 0.5             % m/s
        maxAcceleration = 2.0         % m/s^2
        smoothingFactor = 0.8
        
        % Clinical parameters
        criticalZones = []            % Clinical critical areas
        workflowPriority = 'normal'   % normal, urgent, emergency
        patientSafety = true
        
        % Debug and visualization
        verbose = true
        visualizationEnabled = true
        loggingEnabled = true
    end
    
    methods
        function obj = human_aware_controller(config)
            % Constructor - Initialize human-aware controller
            % Inputs:
            %   config - (optional) Configuration structure
            
            if nargin < 1
                config = struct();
            end
            
            % Apply configuration
            obj.applyConfiguration(config);
            
            % Initialize components
            obj.initializeHumanDetection();
            obj.initializeSafetyZones();
            obj.initializeMotionPlanning();
            obj.initializeSensorFusion();
            obj.initializeClinicalContext();
            obj.initializePerformanceMonitoring();
            
            if obj.verbose
                fprintf('Human-Aware Controller initialized\n');
                fprintf('Detection range: %.1f m\n', obj.detectionRange);
                fprintf('Safety distance: %.2f m\n', obj.minSafetyDistance);
                fprintf('Prediction horizon: %.1f s\n', obj.predictionHorizon);
            end
        end
        
        function applyConfiguration(obj, config)
            % Apply configuration parameters
            
            if isfield(config, 'detectionRange')
                obj.detectionRange = config.detectionRange;
            end
            if isfield(config, 'predictionHorizon')
                obj.predictionHorizon = config.predictionHorizon;
            end
            if isfield(config, 'minSafetyDistance')
                obj.minSafetyDistance = config.minSafetyDistance;
            end
            if isfield(config, 'maxVelocity')
                obj.maxVelocity = config.maxVelocity;
            end
            if isfield(config, 'verbose')
                obj.verbose = config.verbose;
            end
        end
        
        function initializeHumanDetection(obj)
            % Initialize human detection and tracking components
            
            % Human detector using YOLO-based architecture
            obj.humanDetector = struct();
            obj.humanDetector.model = 'yolov8-human';
            obj.humanDetector.confidenceThreshold = 0.7;
            obj.humanDetector.nmsThreshold = 0.4;
            obj.humanDetector.inputSize = [640, 640];
            
            % Pose estimator for human body keypoints
            obj.poseEstimator = struct();
            obj.poseEstimator.model = 'hrnet-w48';
            obj.poseEstimator.numKeypoints = 17;
            obj.poseEstimator.keypointNames = {
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            };
            
            % Intent predictor using LSTM network
            obj.intentPredictor = struct();
            obj.intentPredictor.model = obj.createIntentModel();
            obj.intentPredictor.sequenceLength = 30;  % frames
            obj.intentPredictor.intentClasses = {
                'approaching_robot', 'departing_robot', 'crossing_path',
                'working_nearby', 'observing', 'unpredictable'
            };
            
            % Tracking filter (Extended Kalman Filter)
            obj.trackingFilter = struct();
            obj.trackingFilter.stateDim = 6;  % [x, y, z, vx, vy, vz]
            obj.trackingFilter.measDim = 3;   % [x, y, z]
            obj.trackingFilter.processNoise = 0.1 * eye(6);
            obj.trackingFilter.measurementNoise = 0.05 * eye(3);
        end
        
        function model = createIntentModel(obj)
            % Create LSTM model for human intent prediction
            
            model = struct();
            model.type = 'lstm';
            model.inputSize = 51;  % 17 keypoints * 3 coordinates
            model.hiddenSize = 128;
            model.numLayers = 2;
            model.outputSize = length(obj.intentPredictor.intentClasses);
            model.dropout = 0.3;
            
            % Model architecture details
            model.layers = {
                struct('type', 'lstm', 'hiddenSize', 128, 'numLayers', 1)
                struct('type', 'dropout', 'rate', 0.3)
                struct('type', 'lstm', 'hiddenSize', 64, 'numLayers', 1)
                struct('type', 'dropout', 'rate', 0.3)
                struct('type', 'fullyconnected', 'outputSize', model.outputSize)
                struct('type', 'softmax')
            };
        end
        
        function initializeSafetyZones(obj)
            % Initialize dynamic safety zone management
            
            obj.safetyZones = struct();
            
            % Define zone types with adaptive parameters
            obj.safetyZones.critical = struct(...
                'radius', obj.emergencyThreshold, ...
                'color', [1, 0, 0], ...  % Red
                'velocityScaling', 0.0, ...  % Stop
                'priority', 1);
            
            obj.safetyZones.warning = struct(...
                'radius', obj.minSafetyDistance, ...
                'color', [1, 0.5, 0], ...  % Orange
                'velocityScaling', 0.3, ...
                'priority', 2);
            
            obj.safetyZones.caution = struct(...
                'radius', obj.minSafetyDistance * 1.5, ...
                'color', [1, 1, 0], ...  % Yellow
                'velocityScaling', 0.6, ...
                'priority', 3);
            
            obj.safetyZones.awareness = struct(...
                'radius', obj.detectionRange, ...
                'color', [0, 1, 0], ...  % Green
                'velocityScaling', 0.9, ...
                'priority', 4);
            
            % Initialize history tracking
            obj.zoneHistory = [];
            obj.violationHistory = [];
        end
        
        function initializeMotionPlanning(obj)
            % Initialize advanced motion planning components
            
            % Trajectory optimizer with human constraints
            obj.trajectoryOptimizer = struct();
            obj.trajectoryOptimizer.type = 'constrained_optimization';
            obj.trajectoryOptimizer.objectiveWeights = [
                0.4,  % Time optimality
                0.3,  % Smoothness
                0.2,  % Human comfort
                0.1   % Energy efficiency
            ];
            
            % Constraint generator for human-aware planning
            obj.constraintGenerator = struct();
            obj.constraintGenerator.minDistance = obj.minSafetyDistance;
            obj.constraintGenerator.comfortDistance = obj.minSafetyDistance * 1.2;
            obj.constraintGenerator.socialDistance = obj.minSafetyDistance * 2.0;
            
            % Emergency planner for critical situations
            obj.emergencyPlanner = struct();
            obj.emergencyPlanner.reactionTime = 0.1;  % seconds
            obj.emergencyPlanner.deceleration = 5.0;  % m/s^2
            obj.emergencyPlanner.recoveryStrategies = {
                'immediate_stop', 'retreat_to_safe', 'alternative_path'
            };
        end
        
        function initializeSensorFusion(obj)
            % Initialize multi-modal sensor fusion
            
            % Vision processor for RGB cameras
            obj.visionProcessor = struct();
            obj.visionProcessor.cameraMatrix = eye(3);
            obj.visionProcessor.distortionCoeffs = zeros(1, 5);
            obj.visionProcessor.detectionConfidence = 0.7;
            
            % Depth processor for 3D sensing
            obj.depthProcessor = struct();
            obj.depthProcessor.minDepth = 0.5;  % meters
            obj.depthProcessor.maxDepth = 5.0;   % meters
            obj.depthProcessor.accuracy = 0.01;  % meters
            
            % Thermal processor for enhanced detection
            obj.thermalProcessor = struct();
            obj.thermalProcessor.temperatureRange = [20, 40];  % Celsius
            obj.thermalProcessor.humanTempRange = [36, 37.5];
            obj.thermalProcessor.sensitivity = 0.1;
            
            % Fusion filter (Kalman filter for multi-sensor fusion)
            obj.fusionFilter = struct();
            obj.fusionFilter.stateDim = 6;
            obj.fusionFilter.measDim = 9;  % 3 sensors * 3 coordinates
            obj.fusionFilter.sensorWeights = [0.5, 0.3, 0.2];  % RGB, Depth, Thermal
        end
        
        function initializeClinicalContext(obj)
            % Initialize clinical workflow and safety protocols
            
            % Workflow analyzer for clinical procedures
            obj.workflowAnalyzer = struct();
            obj.workflowAnalyzer.procedures = {
                'medication_preparation', 'patient_administration', 
                'emergency_response', 'routine_check', 'sterile_procedure'
            };
            obj.workflowAnalyzer.currentProcedure = 'routine_check';
            obj.workflowAnalyzer.sensitivityLevel = 'medium';
            
            % Activity recognizer for clinical staff actions
            obj.activityRecognizer = struct();
            obj.activityRecognizer.actions = {
                'reaching_for_medication', 'preparing_injection',
                'documenting', 'patient_interaction', 'sterile_handling'
            };
            obj.activityRecognizer.confidenceThreshold = 0.8;
            
            % Safety protocols for clinical environments
            obj.safetyProtocols = struct();
            obj.safetyProtocols.emergencyStop = true;
            obj.safetyProtocols.sterileZoneProtection = true;
            obj.safetyProtocols.patientPriority = true;
            obj.safetyProtocols.medicationSafety = true;
        end
        
        function initializePerformanceMonitoring(obj)
            % Initialize performance monitoring and metrics
            
            % Safety metrics
            obj.safetyMetrics = struct();
            obj.safetyMetrics.nearMisses = 0;
            obj.safetyMetrics.emergencyStops = 0;
            obj.safetyMetrics.violations = 0;
            obj.safetyMetrics.reactionTimes = [];
            obj.safetyMetrics.minDistances = [];
            
            % Efficiency metrics
            obj.efficiencyMetrics = struct();
            obj.efficiencyMetrics.taskCompletionTimes = [];
            obj.efficiencyMetrics.pathEfficiency = [];
            obj.efficiencyMetrics.velocityProfiles = [];
            obj.efficiencyMetrics.waitingTimes = [];
            
            % Interaction metrics
            obj.interactionMetrics = struct();
            obj.interactionMetrics.humanDetectionRate = [];
            obj.interactionMetrics.intentPredictionAccuracy = [];
            obj.interactionMetrics.trajectorySmoothness = [];
            obj.interactionMetrics.comfortScores = [];
        end
        
        function [humanDetections, poses, intents] = detectHumans(obj, sensorData)
            % Detect humans and estimate their poses and intents
            % Inputs:
            %   sensorData - Structure containing RGB, depth, and thermal data
            % Outputs:
            %   humanDetections - Detected human bounding boxes
            %   poses - Estimated human poses
            %   intents - Predicted human intents
            
            % Process RGB data for human detection
            if isfield(sensorData, 'rgb')
                rgbDetections = obj.processRGBData(sensorData.rgb);
            else
                rgbDetections = [];
            end
            
            % Process depth data for 3D localization
            if isfield(sensorData, 'depth')
                depthDetections = obj.processDepthData(sensorData.depth);
            else
                depthDetections = [];
            end
            
            % Process thermal data for enhanced detection
            if isfield(sensorData, 'thermal')
                thermalDetections = obj.processThermalData(sensorData.thermal);
            else
                thermalDetections = [];
            end
            
            % Fuse detections from multiple sensors
            humanDetections = obj.fuseDetections(rgbDetections, depthDetections, thermalDetections);
            
            % Estimate poses for detected humans
            poses = obj.estimatePoses(humanDetections, sensorData);
            
            % Predict intents based on pose sequences
            intents = obj.predictIntents(poses);
            
            if obj.verbose && ~isempty(humanDetections)
                fprintf('Detected %d humans with poses and intents\n', length(humanDetections));
            end
        end
        
        function detections = processRGBData(obj, rgbImage)
            % Process RGB image for human detection
            
            % Placeholder for YOLO-based human detection
            % In real implementation, this would use deep learning models
            
            detections = [];
            
            % Simulate detection based on image analysis
            [h, w, ~] = size(rgbImage);
            
            % Simulate finding humans in the image
            numHumans = randi([0, 3]);
            for i = 1:numHumans
                % Generate random bounding box
                x = rand() * (w - 100) + 50;
                y = rand() * (h - 200) + 100;
                width = rand() * 50 + 40;
                height = rand() * 100 + 100;
                
                detection = struct();
                detection.bbox = [x, y, width, height];
                detection.confidence = rand() * 0.3 + 0.7;  % 0.7 to 1.0
                detection.class = 'human';
                detection.source = 'rgb';
                
                detections(end+1) = detection;
            end
        end
        
        function detections = processDepthData(obj, depthImage)
            % Process depth image for 3D human localization
            
            detections = [];
            
            % Simulate depth-based detection
            [h, w] = size(depthImage);
            
            % Find human-like shapes in depth data
            numHumans = randi([0, 2]);
            for i = 1:numHumans
                x = rand() * (w - 80) + 40;
                y = rand() * (h - 160) + 80;
                width = rand() * 30 + 30;
                height = rand() * 80 + 80;
                depth = rand() * 3 + 1;  % 1 to 4 meters
                
                detection = struct();
                detection.bbox = [x, y, width, height];
                detection.depth = depth;
                detection.confidence = rand() * 0.2 + 0.6;  % 0.6 to 0.8
                detection.class = 'human';
                detection.source = 'depth';
                
                detections(end+1) = detection;
            end
        end
        
        function detections = processThermalData(obj, thermalImage)
            % Process thermal image for human detection
            
            detections = [];
            
            % Simulate thermal-based detection
            [h, w] = size(thermalImage);
            
            % Find human-temperature regions
            numHumans = randi([0, 1]);
            for i = 1:numHumans
                x = rand() * (w - 60) + 30;
                y = rand() * (h - 120) + 60;
                width = rand() * 20 + 20;
                height = rand() * 60 + 60;
                temperature = rand() * 1.5 + 36;  % 36 to 37.5 Celsius
                
                detection = struct();
                detection.bbox = [x, y, width, height];
                detection.temperature = temperature;
                detection.confidence = rand() * 0.15 + 0.75;  % 0.75 to 0.9
                detection.class = 'human';
                detection.source = 'thermal';
                
                detections(end+1) = detection;
            end
        end
        
        function fusedDetections = fuseDetections(obj, rgbDetections, depthDetections, thermalDetections)
            % Fuse detections from multiple sensors using weighted averaging
            
            allDetections = [rgbDetections, depthDetections, thermalDetections];
            fusedDetections = [];
            
            % Simple fusion based on spatial proximity
            while ~isempty(allDetections)
                % Take first detection as reference
                ref = allDetections(1);
                allDetections(1) = [];
                
                % Find nearby detections
                nearby = [];
                for i = 1:length(allDetections)
                    if obj.bboxOverlap(ref.bbox, allDetections(i).bbox) > 0.3
                        nearby(end+1) = i;
                    end
                end
                
                % Fuse nearby detections
                if isempty(nearby)
                    % No overlap, keep original
                    fusedDetections(end+1) = ref;
                else
                    % Fuse with nearby detections
                    cluster = [ref, allDetections(nearby)];
                    allDetections(nearby) = [];
                    
                    % Weighted average based on confidence and source
                    weights = zeros(1, length(cluster));
                    for i = 1:length(cluster)
                        baseWeight = cluster(i).confidence;
                        sourceWeight = obj.getSourceWeight(cluster(i).source);
                        weights(i) = baseWeight * sourceWeight;
                    end
                    weights = weights / sum(weights);
                    
                    % Compute fused detection
                    fused = struct();
                    fused.bbox = zeros(1, 4);
                    for i = 1:length(cluster)
                        fused.bbox = fused.bbox + weights(i) * cluster(i).bbox;
                    end
                    fused.confidence = max([cluster.confidence]);
                    fused.class = 'human';
                    fused.sources = {cluster.source};
                    
                    fusedDetections(end+1) = fused;
                end
            end
        end
        
        function weight = getSourceWeight(obj, source)
            % Get weight for sensor source based on reliability
            
            switch source
                case 'rgb'
                    weight = 0.5;
                case 'depth'
                    weight = 0.3;
                case 'thermal'
                    weight = 0.2;
                otherwise
                    weight = 0.1;
            end
        end
        
        function overlap = bboxOverlap(obj, bbox1, bbox2)
            % Compute overlap between two bounding boxes
            
            % Convert to [x1, y1, x2, y2] format
            box1 = [bbox1(1), bbox1(2), bbox1(1) + bbox1(3), bbox1(2) + bbox1(4)];
            box2 = [bbox2(1), bbox2(2), bbox2(1) + bbox2(3), bbox2(2) + bbox2(4)];
            
            % Compute intersection
            x1 = max(box1(1), box2(1));
            y1 = max(box1(2), box2(2));
            x2 = min(box1(3), box2(3));
            y2 = min(box1(4), box2(4));
            
            if x2 <= x1 || y2 <= y1
                overlap = 0;
                return;
            end
            
            intersection = (x2 - x1) * (y2 - y1);
            area1 = (box1(3) - box1(1)) * (box1(4) - box1(2));
            area2 = (box2(3) - box2(1)) * (box2(4) - box2(2));
            union = area1 + area2 - intersection;
            
            overlap = intersection / union;
        end
        
        function poses = estimatePoses(obj, detections, sensorData)
            % Estimate human poses from detections
            
            poses = [];
            
            for i = 1:length(detections)
                pose = struct();
                
                % Extract keypoint locations (simplified)
                bbox = detections(i).bbox;
                centerX = bbox(1) + bbox(3) / 2;
                centerY = bbox(2) + bbox(4) / 2;
                
                % Generate 17 keypoints with realistic proportions
                keypoints = zeros(17, 3);  % [x, y, confidence]
                
                % Head keypoints
                keypoints(1, :) = [centerX, bbox(2) + bbox(4) * 0.1, 0.9];  % nose
                keypoints(2, :) = [centerX - bbox(3) * 0.1, bbox(2) + bbox(4) * 0.1, 0.8];  % left eye
                keypoints(3, :) = [centerX + bbox(3) * 0.1, bbox(2) + bbox(4) * 0.1, 0.8];  % right eye
                keypoints(4, :) = [centerX - bbox(3) * 0.15, bbox(2) + bbox(4) * 0.05, 0.7];  % left ear
                keypoints(5, :) = [centerX + bbox(3) * 0.15, bbox(2) + bbox(4) * 0.05, 0.7];  % right ear
                
                % Shoulder keypoints
                keypoints(6, :) = [centerX - bbox(3) * 0.2, bbox(2) + bbox(4) * 0.15, 0.9];  % left shoulder
                keypoints(7, :) = [centerX + bbox(3) * 0.2, bbox(2) + bbox(4) * 0.15, 0.9];  % right shoulder
                
                % Arm keypoints
                keypoints(8, :) = [centerX - bbox(3) * 0.25, bbox(2) + bbox(4) * 0.3, 0.8];  % left elbow
                keypoints(9, :) = [centerX + bbox(3) * 0.25, bbox(2) + bbox(4) * 0.3, 0.8];  % right elbow
                keypoints(10, :) = [centerX - bbox(3) * 0.2, bbox(2) + bbox(4) * 0.5, 0.7];  % left wrist
                keypoints(11, :) = [centerX + bbox(3) * 0.2, bbox(2) + bbox(4) * 0.5, 0.7];  % right wrist
                
                % Hip keypoints
                keypoints(12, :) = [centerX - bbox(3) * 0.15, bbox(2) + bbox(4) * 0.4, 0.8];  % left hip
                keypoints(13, :) = [centerX + bbox(3) * 0.15, bbox(2) + bbox(4) * 0.4, 0.8];  % right hip
                
                % Leg keypoints
                keypoints(14, :) = [centerX - bbox(3) * 0.15, bbox(2) + bbox(4) * 0.6, 0.7];  % left knee
                keypoints(15, :) = [centerX + bbox(3) * 0.15, bbox(2) + bbox(4) * 0.6, 0.7];  % right knee
                keypoints(16, :) = [centerX - bbox(3) * 0.1, bbox(2) + bbox(4) * 0.8, 0.6];  % left ankle
                keypoints(17, :) = [centerX + bbox(3) * 0.1, bbox(2) + bbox(4) * 0.8, 0.6];  % right ankle
                
                pose.keypoints = keypoints;
                pose.confidence = detections(i).confidence;
                pose.bbox = detections(i).bbox;
                pose.timestamp = datetime('now');
                
                poses(end+1) = pose;
            end
        end
        
        function intents = predictIntents(obj, poses)
            % Predict human intents based on pose sequences
            
            intents = [];
            
            for i = 1:length(poses)
                intent = struct();
                
                % Extract features from current pose
                keypoints = poses(i).keypoints;
                
                % Compute simple features for intent prediction
                features = obj.extractIntentFeatures(keypoints);
                
                % Predict intent using simplified logic
                intentScores = obj.computeIntentScores(features);
                
                % Select intent with highest score
                [maxScore, maxIdx] = max(intentScores);
                intent.intent = obj.intentPredictor.intentClasses{maxIdx};
                intent.confidence = maxScore;
                intent.features = features;
                intent.scores = intentScores;
                
                intents(end+1) = intent;
            end
        end
        
        function features = extractIntentFeatures(obj, keypoints)
            % Extract features for intent prediction from keypoints
            
            features = zeros(1, 10);
            
            % Body orientation and movement
            shoulderCenter = (keypoints(6, 1:2) + keypoints(7, 1:2)) / 2;
            hipCenter = (keypoints(12, 1:2) + keypoints(13, 1:2)) / 2;
            bodyVector = hipCenter - shoulderCenter;
            features(1) = atan2(bodyVector(2), bodyVector(1));  % Body orientation
            
            % Arm positions (reaching indicators)
            leftArmExtended = keypoints(10, 2) < keypoints(6, 2);  % Left wrist above shoulder
            rightArmExtended = keypoints(11, 2) < keypoints(7, 2);  % Right wrist above shoulder
            features(2) = leftArmExtended;
            features(3) = rightArmExtended;
            
            % Arm spread (width of workspace)
            armWidth = abs(keypoints(10, 1) - keypoints(11, 1));
            features(4) = armWidth;
            
            % Body lean direction
            bodyLean = shoulderCenter(1) - hipCenter(1);
            features(5) = bodyLean;
            
            % Movement speed (simulated)
            features(6) = rand() * 0.5;  % Placeholder for velocity
            
            % Height indicators
            avgHeight = mean(keypoints(:, 2));
            features(7) = avgHeight;
            
            % Symmetry measures
            leftRightSymmetry = abs(keypoints(10, 2) - keypoints(11, 2));
            features(8) = leftRightSymmetry;
            
            % Workspace occupation
            workspaceArea = obj.computeWorkspaceArea(keypoints);
            features(9) = workspaceArea;
            
            % Confidence in pose estimation
            avgConfidence = mean(keypoints(:, 3));
            features(10) = avgConfidence;
        end
        
        function scores = computeIntentScores(obj, features)
            % Compute intent scores based on features
            
            scores = zeros(1, length(obj.intentPredictor.intentClasses));
            
            % Simplified intent scoring logic
            % In real implementation, this would use the trained LSTM model
            
            % Approaching robot: body oriented forward, arms extended
            if features(1) > 0 && (features(2) || features(3))
                scores(1) = 0.8;
            end
            
            % Departing robot: body oriented away
            if features(1) < 0
                scores(2) = 0.7;
            end
            
            % Crossing path: lateral movement, wide arm spread
            if abs(features(5)) > 0.5 && features(4) > 100
                scores(3) = 0.6;
            end
            
            % Working nearby: moderate arm extension, focused workspace
            if features(4) < 80 && (features(2) || features(3))
                scores(4) = 0.7;
            end
            
            % Observing: no arm extension, stable posture
            if ~features(2) && ~features(3) && features(6) < 0.1
                scores(5) = 0.6;
            end
            
            % Unpredictable: rapid movement, asymmetric posture
            if features(6) > 0.3 || features(8) > 50
                scores(6) = 0.5;
            end
            
            % Normalize scores
            if sum(scores) > 0
                scores = scores / sum(scores);
            else
                scores = ones(1, length(scores)) / length(scores);
            end
        end
        
        function area = computeWorkspaceArea(obj, keypoints)
            % Compute workspace area occupied by human
            
            % Use convex hull of keypoints to estimate area
            validKeypoints = keypoints(keypoints(:, 3) > 0.5, 1:2);
            
            if size(validKeypoints, 1) < 3
                area = 0;
                return;
            end
            
            % Simple area estimation (bounding box)
            minX = min(validKeypoints(:, 1));
            maxX = max(validKeypoints(:, 1));
            minY = min(validKeypoints(:, 2));
            maxY = max(validKeypoints(:, 2));
            
            area = (maxX - minX) * (maxY - minY);
        end
        
        function [trajectory, safetyInfo] = planHumanAwareTrajectory(obj, startPose, goalPose, humanDetections)
            % Plan trajectory with human-aware constraints
            % Inputs:
            %   startPose - Starting robot pose
            %   goalPose - Goal robot pose
            %   humanDetections - Current human detections and intents
            % Outputs:
            %   trajectory - Planned trajectory
            %   safetyInfo - Safety analysis information
            
            % Initialize safety information
            safetyInfo = struct();
            safetyInfo.humanPresent = ~isempty(humanDetections);
            safetyInfo.safetyZones = obj.computeSafetyZones(humanDetections);
            safetyInfo.riskLevel = obj.assessRiskLevel(humanDetections);
            
            % Generate initial trajectory
            initialTrajectory = obj.generateInitialTrajectory(startPose, goalPose);
            
            % Apply human-aware constraints
            constrainedTrajectory = obj.applyHumanConstraints(initialTrajectory, humanDetections);
            
            % Optimize for smoothness and efficiency
            trajectory = obj.optimizeTrajectory(constrainedTrajectory, humanDetections);
            
            % Validate trajectory safety
            [isSafe, violations] = obj.validateTrajectorySafety(trajectory, humanDetections);
            safetyInfo.isSafe = isSafe;
            safetyInfo.violations = violations;
            
            % If not safe, generate alternative trajectory
            if ~isSafe
                [trajectory, safetyInfo] = obj.generateAlternativeTrajectory(startPose, goalPose, humanDetections);
            end
            
            % Update safety metrics
            obj.updateSafetyMetrics(safetyInfo);
            
            if obj.verbose
                fprintf('Human-aware trajectory planned: Risk level = %s\n', safetyInfo.riskLevel);
                fprintf('Safety zones: %d active\n', length(safetyInfo.safetyZones));
            end
        end
        
        function safetyZones = computeSafetyZones(obj, humanDetections)
            % Compute dynamic safety zones around detected humans
            
            safetyZones = [];
            
            for i = 1:length(humanDetections)
                human = humanDetections(i);
                
                % Extract human position (simplified from bounding box)
                bbox = human.bbox;
                humanPos = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                
                % Create safety zones for this human
                for zoneType = fieldnames(obj.safetyZones)'
                    zoneType = zoneType{1};
                    zoneConfig = obj.safetyZones.(zoneType);
                    
                    zone = struct();
                    zone.type = zoneType;
                    zone.center = humanPos;
                    zone.radius = zoneConfig.radius;
                    zone.color = zoneConfig.color;
                    zone.velocityScaling = zoneConfig.velocityScaling;
                    zone.priority = zoneConfig.priority;
                    zone.humanId = i;
                    zone.timestamp = datetime('now');
                    
                    % Adjust zone based on human intent
                    if isfield(human, 'intent')
                        zone = obj.adjustZoneForIntent(zone, human.intent);
                    end
                    
                    safetyZones(end+1) = zone;
                end
            end
        end
        
        function zone = adjustZoneForIntent(obj, zone, intent)
            % Adjust safety zone parameters based on human intent
            
            switch intent.intent
                case 'approaching_robot'
                    % Expand warning and critical zones
                    if strcmp(zone.type, 'warning') || strcmp(zone.type, 'critical')
                        zone.radius = zone.radius * 1.2;
                    end
                    
                case 'crossing_path'
                    % Expand all zones
                    zone.radius = zone.radius * 1.1;
                    
                case 'unpredictable'
                    % Maximize safety margins
                    zone.velocityScaling = zone.velocityScaling * 0.8;
                    
                case 'working_nearby'
                    % Reduce awareness zone (person is stationary)
                    if strcmp(zone.type, 'awareness')
                        zone.radius = zone.radius * 0.8;
                    end
            end
        end
        
        function riskLevel = assessRiskLevel(obj, humanDetections)
            % Assess overall risk level based on human detections and intents
            
            if isempty(humanDetections)
                riskLevel = 'low';
                return;
            end
            
            maxRisk = 0;
            
            for i = 1:length(humanDetections)
                human = humanDetections(i);
                
                % Base risk from distance
                if isfield(human, 'distance')
                    distance = human.distance;
                else
                    distance = 2.0;  % Default distance
                end
                
                if distance < obj.emergencyThreshold
                    risk = 1.0;  % Critical
                elseif distance < obj.minSafetyDistance
                    risk = 0.7;  % High
                elseif distance < obj.minSafetyDistance * 1.5
                    risk = 0.4;  % Medium
                else
                    risk = 0.1;  % Low
                end
                
                % Adjust risk based on intent
                if isfield(human, 'intent')
                    switch human.intent.intent
                        case 'approaching_robot'
                            risk = risk * 1.3;
                        case 'crossing_path'
                            risk = risk * 1.2;
                        case 'unpredictable'
                            risk = risk * 1.5;
                        case 'working_nearby'
                            risk = risk * 0.8;
                    end
                end
                
                maxRisk = max(maxRisk, risk);
            end
            
            % Convert numeric risk to categorical
            if maxRisk >= 0.8
                riskLevel = 'critical';
            elseif maxRisk >= 0.5
                riskLevel = 'high';
            elseif maxRisk >= 0.2
                riskLevel = 'medium';
            else
                riskLevel = 'low';
            end
        end
        
        function trajectory = generateInitialTrajectory(obj, startPose, goalPose)
            % Generate initial trajectory using simple interpolation
            
            nWaypoints = 50;
            trajectory = zeros(nWaypoints, length(startPose));
            
            for i = 1:nWaypoints
                alpha = (i - 1) / (nWaypoints - 1);
                trajectory(i, :) = (1 - alpha) * startPose + alpha * goalPose;
            end
        end
        
        function constrainedTrajectory = applyHumanConstraints(obj, trajectory, humanDetections)
            % Apply human-aware constraints to trajectory
            
            constrainedTrajectory = trajectory;
            
            for i = 1:size(trajectory, 1)
                waypoint = trajectory(i, :);
                
                % Check waypoint against all safety zones
                for j = 1:length(humanDetections)
                    human = humanDetections(j);
                    
                    % Compute distance to human
                    if isfield(human, 'position')
                        humanPos = human.position;
                    else
                        % Extract position from bounding box
                        bbox = human.bbox;
                        humanPos = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                    end
                    
                    waypointPos = waypoint(1:2);  % Assuming x, y are first two elements
                    distance = norm(waypointPos - humanPos);
                    
                    % Apply velocity scaling based on safety zones
                    if distance < obj.emergencyThreshold
                        % Emergency stop - modify trajectory to avoid
                        constrainedTrajectory = obj.avoidWaypoint(constrainedTrajectory, i, humanPos);
                    elseif distance < obj.minSafetyDistance
                        % Warning zone - slow down
                        constrainedTrajectory(i, :) = waypoint * 0.3;
                    elseif distance < obj.minSafetyDistance * 1.5
                        % Caution zone - moderate speed
                        constrainedTrajectory(i, :) = waypoint * 0.6;
                    end
                end
            end
        end
        
        function trajectory = avoidWaypoint(obj, trajectory, waypointIdx, humanPos)
            % Modify trajectory to avoid specific waypoint
            
            % Simple avoidance: add lateral deviation
            avoidanceDistance = 0.3;  % meters
            currentWaypoint = trajectory(waypointIdx, :);
            
            % Compute avoidance direction (perpendicular to human direction)
            humanDirection = humanPos / norm(humanPos);
            avoidanceDirection = [-humanDirection(2), humanDirection(1)];
            
            % Apply avoidance
            currentWaypoint(1:2) = currentWaypoint(1:2) + avoidanceDirection * avoidanceDistance;
            trajectory(waypointIdx, :) = currentWaypoint;
            
            % Smooth surrounding waypoints
            if waypointIdx > 1
                trajectory(waypointIdx-1, :) = 0.7 * trajectory(waypointIdx-1, :) + 0.3 * currentWaypoint;
            end
            if waypointIdx < size(trajectory, 1)
                trajectory(waypointIdx+1, :) = 0.7 * trajectory(waypointIdx+1, :) + 0.3 * currentWaypoint;
            end
        end
        
        function optimizedTrajectory = optimizeTrajectory(obj, trajectory, humanDetections)
            % Optimize trajectory for smoothness and efficiency
            
            % Apply smoothing filter
            optimizedTrajectory = obj.smoothTrajectory(trajectory);
            
            % Optimize timing
            optimizedTrajectory = obj.optimizeTiming(optimizedTrajectory, humanDetections);
        end
        
        function smoothedTrajectory = smoothTrajectory(obj, trajectory)
            % Apply smoothing to trajectory
            
            smoothedTrajectory = trajectory;
            windowSize = 5;
            
            for i = windowSize:(size(trajectory, 1) - windowSize)
                window = trajectory(i-windowSize+1:i+windowSize-1, :);
                smoothedTrajectory(i, :) = mean(window, 1);
            end
        end
        
        function optimizedTrajectory = optimizeTiming(obj, trajectory, humanDetections)
            % Optimize timing based on human presence
            
            optimizedTrajectory = trajectory;
            nWaypoints = size(trajectory, 1);
            
            for i = 1:nWaypoints
                waypoint = trajectory(i, :);
                waypointPos = waypoint(1:2);
                
                % Find minimum distance to any human
                minDistance = inf;
                for j = 1:length(humanDetections)
                    if isfield(humanDetections(j), 'position')
                        humanPos = humanDetections(j).position;
                    else
                        bbox = humanDetections(j).bbox;
                        humanPos = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                    end
                    distance = norm(waypointPos - humanPos);
                    minDistance = min(minDistance, distance);
                end
                
                % Adjust timing based on distance
                if minDistance < obj.emergencyThreshold
                    % Add wait time
                    optimizedTrajectory = [optimizedTrajectory(1:i, :); waypoint; optimizedTrajectory(i:end, :)];
                elseif minDistance < obj.minSafetyDistance
                    % Slow down segment
                    if i < nWaypoints
                        optimizedTrajectory(i:i+1, :) = repmat(waypoint, 2, 1);
                    end
                end
            end
        end
        
        function [isSafe, violations] = validateTrajectorySafety(obj, trajectory, humanDetections)
            % Validate trajectory safety against human constraints
            
            isSafe = true;
            violations = [];
            
            for i = 1:size(trajectory, 1)
                waypoint = trajectory(i, :);
                waypointPos = waypoint(1:2);
                
                for j = 1:length(humanDetections)
                    human = humanDetections(j);
                    
                    % Get human position
                    if isfield(human, 'position')
                        humanPos = human.position;
                    else
                        bbox = human.bbox;
                        humanPos = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                    end
                    
                    distance = norm(waypointPos - humanPos);
                    
                    % Check for violations
                    if distance < obj.emergencyThreshold
                        isSafe = false;
                        violation = struct();
                        violation.type = 'critical_distance';
                        violation.waypoint = i;
                        violation.humanId = j;
                        violation.distance = distance;
                        violations(end+1) = violation;
                    elseif distance < obj.minSafetyDistance
                        violation = struct();
                        violation.type = 'safety_distance';
                        violation.waypoint = i;
                        violation.humanId = j;
                        violation.distance = distance;
                        violations(end+1) = violation;
                    end
                end
            end
        end
        
        function [trajectory, safetyInfo] = generateAlternativeTrajectory(obj, startPose, goalPose, humanDetections)
            % Generate alternative trajectory when primary is unsafe
            
            % Try different approach strategies
            strategies = {'go_around', 'wait_and_proceed', 'alternative_path'};
            
            for i = 1:length(strategies)
                switch strategies{i}
                    case 'go_around'
                        trajectory = obj.planGoAroundTrajectory(startPose, goalPose, humanDetections);
                    case 'wait_and_proceed'
                        trajectory = obj.planWaitTrajectory(startPose, goalPose, humanDetections);
                    case 'alternative_path'
                        trajectory = obj.planAlternativePath(startPose, goalPose, humanDetections);
                end
                
                % Validate new trajectory
                [isSafe, violations] = obj.validateTrajectorySafety(trajectory, humanDetections);
                if isSafe
                    safetyInfo = struct();
                    safetyInfo.isSafe = true;
                    safetyInfo.strategy = strategies{i};
                    safetyInfo.violations = [];
                    return;
                end
            end
            
            % If all strategies fail, emergency stop
            trajectory = repmat(startPose, 10, 1);
            safetyInfo = struct();
            safetyInfo.isSafe = false;
            safetyInfo.strategy = 'emergency_stop';
            safetyInfo.violations = violations;
        end
        
        function trajectory = planGoAroundTrajectory(obj, startPose, goalPose, humanDetections)
            % Plan trajectory that goes around humans
            
            % Compute centroid of human positions
            humanCentroid = zeros(1, 2);
            for i = 1:length(humanDetections)
                if isfield(humanDetections(i), 'position')
                    humanCentroid = humanCentroid + humanDetections(i).position;
                else
                    bbox = humanDetections(i).bbox;
                    humanCentroid = humanCentroid + [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                end
            end
            humanCentroid = humanCentroid / length(humanDetections);
            
            % Create waypoints that avoid the centroid
            avoidanceRadius = obj.minSafetyDistance * 2;
            avoidanceAngle = atan2(humanCentroid(2), humanCentroid(1)) + pi/2;
            avoidancePoint = humanCentroid + avoidanceRadius * [cos(avoidanceAngle), sin(avoidanceAngle)];
            
            % Generate three-segment trajectory
            midPose = goalPose;
            midPose(1:2) = avoidancePoint;
            
            % Interpolate segments
            segment1 = obj.interpolateSegment(startPose, midPose, 20);
            segment2 = obj.interpolateSegment(midPose, goalPose, 20);
            
            trajectory = [segment1; segment2];
        end
        
        function trajectory = planWaitTrajectory(obj, startPose, goalPose, humanDetections)
            % Plan trajectory with waiting periods
            
            % Estimate wait time based on human movement
            waitTime = 2.0;  % seconds
            waitWaypoints = round(waitTime / obj.timeStep);
            
            % Create trajectory with pause
            initialSegment = obj.interpolateSegment(startPose, startPose, waitWaypoints);
            finalSegment = obj.interpolateSegment(startPose, goalPose, 30);
            
            trajectory = [initialSegment; finalSegment];
        end
        
        function trajectory = planAlternativePath(obj, startPose, goalPose, humanDetections)
            % Plan completely different path
            
            % Generate waypoints that maximize distance from humans
            nWaypoints = 40;
            trajectory = zeros(nWaypoints, length(startPose));
            
            for i = 1:nWaypoints
                alpha = (i - 1) / (nWaypoints - 1);
                baseWaypoint = (1 - alpha) * startPose + alpha * goalPose;
                
                % Add lateral offset to avoid humans
                maxOffset = 0.5;
                offset = maxOffset * sin(alpha * pi);
                baseWaypoint(1) = baseWaypoint(1) + offset;
                
                trajectory(i, :) = baseWaypoint;
            end
        end
        
        function segment = interpolateSegment(obj, startPose, endPose, nWaypoints)
            % Interpolate between two poses
            
            segment = zeros(nWaypoints, length(startPose));
            
            for i = 1:nWaypoints
                alpha = (i - 1) / (nWaypoints - 1);
                segment(i, :) = (1 - alpha) * startPose + alpha * endPose;
            end
        end
        
        function updateSafetyMetrics(obj, safetyInfo)
            % Update safety metrics based on trajectory planning
            
            if isfield(safetyInfo, 'violations')
                obj.safetyMetrics.violations = obj.safetyMetrics.violations + length(safetyInfo.violations);
                
                % Categorize violations
                for i = 1:length(safetyInfo.violations)
                    violation = safetyInfo.violations(i);
                    switch violation.type
                        case 'critical_distance'
                            obj.safetyMetrics.nearMisses = obj.safetyMetrics.nearMisses + 1;
                        case 'safety_distance'
                            % Count as minor violation
                        case 'emergency_stop'
                            obj.safetyMetrics.emergencyStops = obj.safetyMetrics.emergencyStops + 1;
                    end
                end
            end
            
            % Log safety information
            if obj.loggingEnabled
                obj.logSafetyEvent(safetyInfo);
            end
        end
        
        function logSafetyEvent(obj, safetyInfo)
            % Log safety event for analysis
            
            logEntry = struct();
            logEntry.timestamp = datetime('now');
            logEntry.riskLevel = safetyInfo.riskLevel;
            logEntry.isSafe = safetyInfo.isSafe;
            logEntry.humanPresent = safetyInfo.humanPresent;
            
            if isfield(safetyInfo, 'violations')
                logEntry.violationCount = length(safetyInfo.violations);
            else
                logEntry.violationCount = 0;
            end
            
            % Store in metrics (simplified logging)
            if obj.loggingEnabled
                % In real implementation, write to log file
                if obj.verbose
                    fprintf('Safety event logged: %s, Safe: %s, Violations: %d\n', ...
                        logEntry.riskLevel, string(logEntry.isSafe), logEntry.violationCount);
                end
            end
        end
        
        function metrics = getPerformanceMetrics(obj)
            % Get current performance metrics
            
            metrics = struct();
            metrics.safety = obj.safetyMetrics;
            metrics.efficiency = obj.efficiencyMetrics;
            metrics.interaction = obj.interactionMetrics;
            
            % Compute derived metrics
            if ~isempty(obj.safetyMetrics.minDistances)
                metrics.safety.averageMinDistance = mean(obj.safetyMetrics.minDistances);
            else
                metrics.safety.averageMinDistance = inf;
            end
            
            if ~isempty(obj.safetyMetrics.reactionTimes)
                metrics.safety.averageReactionTime = mean(obj.safetyMetrics.reactionTimes);
            else
                metrics.safety.averageReactionTime = 0;
            end
            
            % Safety score (higher is better)
            safetyScore = 1.0;
            safetyScore = safetyScore - obj.safetyMetrics.violations * 0.1;
            safetyScore = safetyScore - obj.safetyMetrics.emergencyStops * 0.2;
            safetyScore = safetyScore - obj.safetyMetrics.nearMisses * 0.05;
            metrics.safety.safetyScore = max(0, safetyScore);
        end
        
        function visualizeSafetyZones(obj, humanDetections, trajectory)
            % Visualize safety zones and trajectory
            
            if ~obj.visualizationEnabled
                return;
            end
            
            figure('Position', [100, 100, 800, 600]);
            hold on;
            
            % Plot safety zones for each human
            for i = 1:length(humanDetections)
                human = humanDetections(i);
                
                % Get human position
                if isfield(human, 'position')
                    humanPos = human.position;
                else
                    bbox = human.bbox;
                    humanPos = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
                end
                
                % Plot safety zones
                for zoneType = fieldnames(obj.safetyZones)'
                    zoneType = zoneType{1};
                    zoneConfig = obj.safetyZones.(zoneType);
                    
                    circle = rectangle('Position', ...
                        [humanPos(1) - zoneConfig.radius, humanPos(2) - zoneConfig.radius, ...
                         2 * zoneConfig.radius, 2 * zoneConfig.radius], ...
                        'Curvature', [1, 1], ...
                        'FaceColor', zoneConfig.color, ...
                        'FaceAlpha', 0.3, ...
                        'EdgeColor', 'none');
                end
                
                % Plot human
                plot(humanPos(1), humanPos(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
                text(humanPos(1), humanPos(2) + 0.1, sprintf('Human %d', i), ...
                    'HorizontalAlignment', 'center');
            end
            
            % Plot trajectory
            if ~isempty(trajectory)
                plot(trajectory(:, 1), trajectory(:, 2), 'b-', 'LineWidth', 2);
                plot(trajectory(1, 1), trajectory(1, 2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
                plot(trajectory(end, 1), trajectory(end, 2), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
            end
            
            % Formatting
            axis equal;
            grid on;
            xlabel('X Position (m)');
            ylabel('Y Position (m)');
            title('Human-Aware Trajectory Planning');
            legend('Critical Zone', 'Warning Zone', 'Caution Zone', 'Awareness Zone', ...
                   'Human', 'Trajectory', 'Start', 'Goal', 'Location', 'best');
            
            xlim([-2, 2]);
            ylim([-2, 2]);
            hold off;
        end
    end
end
