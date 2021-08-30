clear all; close all force; clc;

data = xlsread("DatosOutCompletosTarea.xlsx");
x = data(:,1:9);
y = data(:,10);


testingDataPercent = 20;%solo se usa si NumSetsCrossValidation = 1;
NumSetsCrossValidation = 4  ;
mu = 0.1;
Emin = 0.1; %Peor error aceptable
seed = 2;
IterationsMax = 10e3;
Weight_max = 10;
Weight_min = -10;
NeuronsPerLayer = [20,10,5];% Solo se refiere a las ocultas.Si [] , es una sola neurona, 

[Folds,Crossval,foldW,foldbias] = Perceptron_Multilayer_new(x,y,NeuronsPerLayer,mu,Emin,IterationsMax,NumSetsCrossValidation,testingDataPercent,Weight_min,Weight_max,seed);