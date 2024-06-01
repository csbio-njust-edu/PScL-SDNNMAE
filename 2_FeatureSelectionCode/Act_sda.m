close all, clear all, clc;
addpath('./lib');

log_folder = "./data/log_files"
if ~exist( log_folder,'dir')
    mkdir(log_folder);
end

K = 10;
for i = 0:(K-1)
    fprintf('%d\n', i);
    read_dir = "./data/path" + num2str(i);
    write_dir = read_dir;
    if ~exist(write_dir + "/sda",'dir')
        mkdir(write_dir + "/sda");
    end
    if ~exist(write_dir + "/sda/train",'dir')
        mkdir(write_dir + "/sda/train");
    end
    if ~exist(write_dir + "/sda/val",'dir')
        mkdir(write_dir + "/sda/val");
    end
    sda_single_set(read_dir, write_dir);

end

