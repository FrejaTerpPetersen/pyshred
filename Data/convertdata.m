
clear,clc
filename = "CYLINDER_ALL.mat";
data = load(filename);

% Get field names (variables) in the .mat file
fields = fieldnames(data);

%  Check if the file has at least one variable
if isempty(fields)
    error('The .mat file does not contain any variables.');
end

for i=1:length(fields)
    % Loop over variables to export to csv
    varName = fields{i};
    varData = data.(varName);
    
    % Check if data is numeric or a table
    if isnumeric(varData) || istable(varData)
        % Create a CSV file name based on the .mat file
        %[~, name, ~] = fileparts(filename);
        csvFileName = strcat('cylinder/', varName, '.csv');
    
        % Write to CSV
        if istable(varData)
            writetable(varData, csvFileName);
        else
            writematrix(varData, csvFileName);
        end
    
        fprintf('Data saved to %s\n', csvFileName);
    else
        error('The variable %s is not a numeric array or table.', varName);
    end
end