import_name_data;


velocity_output = calc_velocity(ssid_rank_data);
save -mat7-binary 'velocity_data.mat' 'velocity_output';
