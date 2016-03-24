function [] = display_progress(message,isOctave)
if nargin < 2
    isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0; % checking if octave
end
disp(message);
if isOctave
    fflush(stdout);
end
end