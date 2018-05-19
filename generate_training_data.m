% Generate data for learning fault diagnosis neural network


size = 25;


% observer_dataset_file = strcat('observer_dataset_', strrep(strrep(strrep(datestr(now) ,'-','_'),':','_'),' ','_') , '.csv');
for j=28:1:28
for i=1:1:size
    fault_type = j
    fault_time = randi([200 1400])
    [observed_data] = run_model(fault_type, fault_time);
    observer_dataset_file = strcat('observer_dataset_', strrep(strrep(strrep(datestr(now) ,'-','_'),':','_'),' ','_') , '.csv');
    dlmwrite(observer_dataset_file, observed_data','-append')
    
end
end
