clear all; close all force; clc;

data = xlsread("DatosOutCompletosTarea.xlsx");
x = data(:,1:9);
y = data(:,10);

caso = 1

switch caso
    case 1
    testingDataPercent = 20;%solo se usa si NumSetsCrossValidation = 1;
    NumSetsCrossValidation = 4 ;
    mu = 0.1;
    Emin = 0.03; %Peor error aceptable
    seed = 2;
    IterationsMax = 1e3;
    Weight_max = 10;
    Weight_min = -10;
    NeuronsPerLayer = [20,10,5];% Solo se refiere a las ocultas.Si [] , es una sola neurona, 

    [Folds,Crossval,foldW,foldbias] = Perceptron_Multilayer_new(x,y,NeuronsPerLayer,mu,Emin,IterationsMax,NumSetsCrossValidation,testingDataPercent,Weight_min,Weight_max,seed);

    case 2
    testingDataPercent = 20;%solo se usa si NumSetsCrossValidation = 1;
    NumSetsCrossValidation = 1 ;
    mu = 0.1;
    Emin = 0.001; %Peor error aceptable
    seed = 2;
    IterationsMax = 10e3;
    Weight_max = 10;
    Weight_min = -10;
    NeuronsPerLayer = [30,20,10,10,5];% Solo se refiere a las ocultas.Si [] , es una sola neurona, 

    [Folds,Crossval,foldW,foldbias] = Perceptron_Multilayer_new(x,y,NeuronsPerLayer,mu,Emin,IterationsMax,NumSetsCrossValidation,testingDataPercent,Weight_min,Weight_max,seed);

    case 3
    testingDataPercent = 20;%solo se usa si NumSetsCrossValidation = 1;
    NumSetsCrossValidation = 1 ;
    mu = 0.1;
    Emin = 0.001; %Peor error aceptable
    seed = 2;
    IterationsMax = 50e3;
    Weight_max = 10;
    Weight_min = -10;
    NeuronsPerLayer = [64,32,16,18,4,2];% Solo se refiere a las ocultas.Si [] , es una sola neurona, 

    [Folds,Crossval,foldW,foldbias] = Perceptron_Multilayer_new(x,y,NeuronsPerLayer,mu,Emin,IterationsMax,NumSetsCrossValidation,testingDataPercent,Weight_min,Weight_max,seed);

    otherwise
        1;
end
