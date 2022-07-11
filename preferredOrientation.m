function [PO,OSI] = preferredOrientation(orientations,tuningCurves)
% function preferredOrientation - calculates orientation statistics
%
%   [PO,OSI] = preferredOrientation(OD,TC) will calculate the preferred
%   orientation (PO) and orientation selectivity index (OSI). The input
%   OD must be a vector of orientations (in degrees). TC must be
%   a tuning curve array with size(TC,1) = length(OD). The outputs are 
%   vectors with size 1 x size(TC,2).
%

arguments
   orientations (:,1) double  
   tuningCurves double {mustMatchRows(orientations,tuningCurves)}
end

% Caclulate orientation selectivity
orientationsRad = orientations*pi/180;
z = sum(tuningCurves.*exp(2*1i*orientationsRad));
OSI = abs(z)./sum(tuningCurves);

% Caclulate preferred orientation
POrad =  0.5*angle(z);
PO = POrad*180/pi;
PO(PO<0) = PO(PO<0) + 180;

end

% This function validates that inputs have the same number of rows. It
% throws an error if the number of rows don't match
function mustMatchRows(a,b)
    if size(a,1) ~= size(b,1)
        throwAsCaller(MException('QBW:RowMismatch', ...
            'Number of orientations must match row size of tuning curves'));
    end
end

