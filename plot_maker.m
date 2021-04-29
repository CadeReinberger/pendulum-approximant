clc;clear;clf;

PLOT_VAL = 'traj'; %'traj' or 'per'
PLOT_TYPE = 'norm'; %'norm', 'rel', or 'abs'
NS = [5 10]; %list of integer values of N. 

PLOTRESRAT = 10; %skipped amount in exact solutions

if strcmp(PLOT_TYPE, 'norm')
    system(join(['python get_plot_data.py ' PLOT_VAL PLOT_TYPE join(string(NS))]));
    for n = NS
        data = readtable(['data/data_' int2str(n) '.csv']);
        if strcmp(PLOT_VAL, 'traj')
            times = data.times;
            nths = data.num_thetas;
            sths = data.series_thetas;
            aths = data.approximant_thetas;
            a = plot(times,aths,'-','linewidth',2,'DisplayName',strcat('A', int2str(n))); hold on;% plot the approximant
            s = plot(times,sths,'--','linewidth',2,'DisplayName',strcat('S', int2str(n))); %plot the taylor series
            e = plot(times(1:PLOTRESRAT:end),nths(1:PLOTRESRAT:end),'.','markersize',7,'DisplayName',strcat('E', int2str(n))); %plot the Euler's method
        else
            thetas = data.thetas;
            ets = data.exact_periods;
            sts = data.series_periods;
            ats = data.approximant_periods;
            a = plot(thetas,ats,'-','linewidth',2,'DisplayName',strcat('A', int2str(n))); hold on;% plot the approximant
            s = plot(thetas,sts,'--','linewidth',2,'DisplayName',strcat('S', int2str(n))); %plot the taylor series
            e = plot(thetas(1:PLOTRESRAT:end),ets(1:PLOTRESRAT:end),'.','markersize',7,'DisplayName',strcat('E', int2str(n))); %plot the Euler's method
        end
    end
else
        system(join(['python get_plot_data.py ' PLOT_VAL PLOT_TYPE join(string(NS))]));
    for n = NS
        data = readtable(['data/data_' int2str(n) '.csv']);
        ts = data.ts;
        ses = data.series_errors;
        aes = data.approximant_errors;
        s = plot(ts,ses,'--','linewidth',2,'DisplayName',strcat('S', int2str(n))); hold on; %plot the taylor series
        a = plot(ts,aes,'-','linewidth',2,'DisplayName',strcat('A', int2str(n))); % plot the approximant
    end 
end