close all; clear all;

%Parametros
data = xlsread("DatosOutCompletosTarea.xlsx");
x = data(:,1:9);
y = data(:,10);
testingDataPercent = 20;
Weight_max = 10;
Weight_min = -10;

% NumSetsCrossValidation = 4;
mu = 0.05;  %Si mu es muy bajo el error tiende a ser constante porque no varía mucho los pesos
Emin = 0.05; %max error aceptable
seed = 1;
IterationsMax = 1000;
tipo = 'p';
tmax = 2;

%Normalizacion
x = x/norm(x);
x = [x,ones(size(x,1),1)];

%Semilla
s = RandStream('mlfg6331_64',"Seed",seed); 

%Inicializacion datos para graficar salida de iteraciones y errores
%obtenidos
% Iterations = zeros(NumSetsCrossValidation,1);
% ErrorTraining = zeros(NumSetsCrossValidation,1);
% ErrorTesting = zeros(NumSetsCrossValidation,1);
% IterBolsillo = zeros(NumSetsCrossValidation,1);

%Definicion de datos de prueba y entrenamiento
%index = crossvalind('Kfold', size(x,1), NumSetsCrossValidation);
%for i = 1:NumSetsCrossValidation % toma la parte i-ésima como muestra de prueba y las otras partes como muestra de entrenamiento
%    test = (index == i);
%    train = ~test;
%    x_training = x(train, :);
%    x_testing = x(test, :);
%    y_training = y(train);
%    y_testing = y(test);

    % Particion Entranamiento / Prueba
    index = randsample(s,1:size(x,1),round(testingDataPercent*size(x,1)/100));
    x_testing = x(index,:);
    y_testing = y(index);
    x_training = x(setdiff(1:end,index),:);
    y_training = y(setdiff(1:end,index));


    %Inicializacion
    Weight2 = Weight_min+(Weight_max-Weight_min)*rand(s,1,size(x_training,2)); %W{layer}(neurona capa previa,neurona capa posterior)
    %Weight = 2*rand(s,1,size(x_training,2))-1; %rango entre -1 y 1
    W_bolsillo = Weight;
    
    %Training
    iteration = 0;
    Errores = [];
    ErroresTesting = [];
    E = Emin + 1;
    t=0;
    while iteration < IterationsMax && E > Emin
        iteration = iteration + 1;
        k = randi(s,size(x_training,1)); %indice de los datos de entrada k
        %k = mod(iteration,size(x_training,1))+1;
        weighted_sum = Weight * x_training(k,:)';
        if strcmp(tipo,'LMS')
            output = weighted_sum;
            error = output - y_training(k);
            dWeight = mu * error * x_training(k,:);
            Weight = Weight - dWeight;
        else 
            z_k = weighted_sum*y_training(k);
            if sign(z_k) < 0
                Weight = Weight + mu*(y_training(k)*x_training(k,:)')';
                t = 0;
            else
                t=t+1;
                if t > tmax
                    tmax = t;
                    W_bolsillo = Weight;
                    %IterBolsillo(i,iteration) = 1;
                end
            end
        end
        
        z = Weight*x_training';
        z = sign(z); %Umbral igual para LMS y perceptron por clases -1 y 1
        cantidadErrores = sum(y_training~=z');
        E = sum((y_training - z').^2)/2;
        maxErrores = sum((2*ones(size(y_training))).^2)/2;
        E = E / maxErrores;
        Errores = [Errores;E];
                
        %Error de prueba
        z_testing = Weight * x_testing';
        z_testing = sign(z_testing);
        E_testing = sum(y_testing~=z_testing');
        E_testing = E_testing / (2*size(x_testing,1));
        ErroresTesting = [ErroresTesting; E_testing];
    end
    figure
    plot(Errores,'b');
    hold on;
    plot(ErroresTesting,'r');
    hold off; 
    
    if E > Emin
        Weight = W_bolsillo;
        disp('Nos quedamos con bolsillo')
    end
    
    %Testing
    z = Weight * x_testing';
    z(z>=0) = 1;
    z(z<0) = -1;
    ErroresTotales = sum(y_testing~=z');
    PorcentajeError = ErroresTotales*100/size(y_testing,1);
%     
%     Iterations(i) = iteration;
%     ErrorTraining(i) = E;
%     ErrorTesting(i) = E_testing;
% end
% Fold = [1:NumSetsCrossValidation]';
% table = table(Fold,Iterations,ErrorTraining,ErrorTesting)