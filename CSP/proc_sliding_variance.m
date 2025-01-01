function dat = proc_variance(dat, windowSize, numWindows, calcStd)
%PROC_VARIANCE_SLIDING - computes the variance using sliding windows
%
%Synopsis
% dat = proc_variance_sliding(dat, windowSize, numWindows, <calcStd=0>)
%
% IN   dat        - data structure of continuous or epoched data
%      windowSize - size of the sliding window (in samples)
%      numWindows - total number of windows
%      calcStd    - standard deviation is calculated instead of variance
%
% OUT  dat        - updated data structure with sliding window results
%
% Description
% This function calculates the variance (or standard deviation) using a
% sliding window approach with a specified window size and number of windows.

if nargin < 3
    error('Not enough input arguments. Provide dat, windowSize, and numWindows.');
end
if nargin < 4
    calcStd = 0; % Default is to calculate variance
end

misc_checkType(dat, 'STRUCT(x)');
misc_checkTypeIfExists('windowSize', 'INT');
misc_checkTypeIfExists('numWindows', 'INT');
misc_checkTypeIfExists('calcStd', 'BOOL');
dat = misc_history(dat);

[T, nChans, nMotos] = size(dat.x);

if windowSize > T
    error('Window size cannot be larger than the total number of samples.');
end
if numWindows < 1
    error('numWindows must be greater than 0.');
end

% Step size calculation
if numWindows == 1
    stepSize = T - windowSize; % Special case: one window spans most of the data
else
    stepSize = (T - windowSize) / (numWindows - 1);
    stepSize = floor(stepSize); % Ensure stepSize is an integer
end

% Initialize the output variables
xo = zeros(numWindows, nChans, nMotos); % Result matrix
dat.t = zeros(1, numWindows); % Store time indices for each window

% Perform sliding window computation
for w = 1:numWindows
    % Define window range
    startIdx = (w - 1) * stepSize + 1;
    endIdx = startIdx + windowSize - 1;
    if endIdx > T
        endIdx = T; % Prevent index out of bounds
        startIdx = endIdx - windowSize + 1; % Adjust startIdx for last window
    end
    Ti = startIdx:endIdx;
    
    % Calculate variance or standard deviation
    if calcStd
        xo(w, :, :) = reshape(std(dat.x(Ti, :, :), 0, 1), [1, nChans, nMotos]);
    else
        xo(w, :, :) = reshape(var(dat.x(Ti, :, :), 0, 1), [1, nChans, nMotos]);
    end
    
    % Store the last time index for the window
    dat.t(w) = Ti(end);
end

% Update the data structure
dat.x = xo;

fprintf('Step size calculated: %d samples\n', stepSize);
