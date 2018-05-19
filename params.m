%Параметры модели
%
%Pipes:
pipe_diameter=2;%(m)
pipe_length_short=5;%(m)
initial_pressure=0;%(Pa)
pipe_length_long=50;%(m)
pump_time_delay=5;
length_of_local_resistances=3.5;

%Car_Tank:
fl_level_car_tank=100;
initial_fluid_volume=1000;%(m^3)
pressurization=100000;%(Pa)
tank_cross_section_area_big=50;%(m^2)
%miniTank
pressurization_mini=0;%(pa)
portb_height=5;%(m)

%System temperature
temp=60;

%2-way directional valve
%Valve passage maximum area:(m^2)
passage_max_area=pi*(pipe_diameter/2)^2;
passage_max_area_big_pipe=pi*(pipe_diameter)^2;
valve_stroke=0.05;
valve_gain=0.01;
cutoff_time_const=0.1;
min_vol=20;
%Valve maximum opening
valve_max_opening=pipe_diameter;

area=1;%(m^2)
mini_pump_rps=120;
pump_speed=20000;
gain_coef=4000*10;