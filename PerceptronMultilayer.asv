close all; clear all;

%Parametros
data = xlsread("DatosOutCompletosTarea.xlsx");
x = data(:,1:9);
y = data(:,10);
%testingDataPercent = 20;
NumSetsCrossValidation = 4;
mu = 0.5;
Emin = 0.1; %Peor error aceptable
seed = 2;
IterationsMax = 1e3;
L = 2; %Número de capas ocultas
NeuronsPerLayer = [3 2];
Weight_max = 10;
Weight_min = -10;

assert(size(NeuronsPerLayer,2) == L);

%Normalizacion
x = x/norm(x);

%Semilla
s = RandStream('mlfg6331_64',"Seed",seed); 

%Inicializacion datos para graficar salida de iteraciones y errores
%obtenidos
Iterations = zeros(NumSetsCrossValidation,1);
ErrorTraining = zeros(NumSetsCrossValidation,1);
ErrorTesting = zeros(NumSetsCrossValidation,1);

%Definicion de datos de prueba y entrenamiento
index = crossvalind('Kfold', size(x,1), NumSetsCrossValidation);
for j = 1:NumSetsCrossValidation % toma la parte i-ésima como muestra de prueba y las otras 3 partes como muestra de entrenamiento
    test = (index == j);
    train = ~test;
    x_training = x(train, :);
    x_testing = x(test, :);
    y_training = y(train);
    y_testing = y(test);
    % index = randsample(s,1:size(x,1),round(testingDataPercent*size(x,1)/100));
    % x_testing = x(index,:);
    % y_testing = y(index);
    % x_training = x(setdiff(1:end,index),:);
    % y_training = y(setdiff(1:end,index));
    
    %Inicializacion
    NeuronsPerLayer = [size(x_training,2),NeuronsPerLayer,1]; %Agrega capa inicial de datos de entrada y capa final de salida
    L = L+2; %Capas ocultas de entrada más 2 de datos iniciales y salida
    weights = {[]};
    bias = {[]};
    for l = 1:L-1
        weights{end+1} = Weight_min+(Weight_max-Weight_min)*rand(s,NeuronsPerLayer(l),NeuronsPerLayer(l+1)); %W{layer}(neurona capa previa,neurona capa posterior) 
        bias{end+1} = rand(s,1,NeuronsPerLayer(l+1)); %bias por capa
    end
    iterations = 0;
    E=Emin+1;
    Errores=[];
    while E>Emin && iterations < IterationsMax
        iterations = iterations + 1;
        %fprintf([num2str(iterations) ' ']);
        %k = randi(size(x_training,1)); %indice de los datos de entrada k
        k = mod(iterations,size(x_training,1))+1;
        a = {[x_training(k,:)]};
        z = {[]};
        
        %Feedforward
        for l = 2:L
            z{end+1} = weights{l}'*a{l-1}'+bias{l}';
            if l == L
                a{end+1} = z{end};
            else
                a{end+1} = sigmoideFunction(z{end});
            end
        end
        
        %Output Error
        delta = cell(1,L);
        delta{L} = (a{L}-y_training(k))*dsigmoideFunction(z{L});
        
        %Backpropagation
        for l = L-1:-1:2
            delta{l} = weights{l+1}*delta{l+1}.*dsigmoideFunction(z{l})';
        end
        
        for l=L:-1:2
            weights{l} = weights{l} - mu * a{l-1}'* delta{l}';
            bias{l} = bias{l} - mu * delta{l}';
        end
        
        output = [];
        %Calculo de la salida:
        for i=1:size(x_training,1)
            a = {[x_training(i,:)]};
            z = {[]};
            %Feedforward
            for l = 2:L
                z{end+1} = weights{l}'*a{l-1}'+bias{l}';
                if l == L
                    a{end+1} = z{end};
                else
                    a{end+1} = sigmoideFunction(z{end});
                end
            end
            output = [output,a{end}];
        end
        output = sign(output);
        cantidadErrores = sum(y_training~=output');
        E = sum((y_training - output').^2)/2;
        maxErrores = sum((2*ones(size(y_training))).^2)/2;
        E = E / maxErrores;
        Errores = [Errores;E];
        
        
    end
    figure
    plot(Errores);
    Iterations(j) = iterations;
    ErrorTraining(j) = E;
end
Fold = [1:NumSetsCrossValidation]';
table = table(Fold,Iterations,ErrorTraining,ErrorTesting)

