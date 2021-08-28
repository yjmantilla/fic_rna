close all; clear all;clc;

dt = datestr(now,'yyyy_mmmm_dd_HH_MM_SS_FFF');
%Parametros
data = xlsread("DatosOutCompletosTarea.xlsx");
x = data(:,1:9);
y = data(:,10);
%testingDataPercent = 20;
Weight_max = 1;
Weight_min = -1;

NumSetsCrossValidation = 10;
mu = 0.0025;  %Si mu es muy bajo el error tiende a ser constante porque no varía mucho los pesos
Emin = 0.05; %max error aceptable
seed = 1;
IterationsMax = 100000;
tipo = 'LMS';
tmax = 2; % NUmero minimo de iteraciones sin equivocarse para actualizar WBolsillo

%Normalizacion
x = x/norm(x);
x = [x,ones(size(x,1),1)];

%Semilla
s = RandStream('mlfg6331_64',"Seed",seed); 

%Inicializacion datos para graficar salida de iteraciones y errores
%obtenidos
Iterations = zeros(NumSetsCrossValidation,1);
ErrorTraining = zeros(NumSetsCrossValidation,1);
ErrorTesting = zeros(NumSetsCrossValidation,1);
IterBolsillo = zeros(NumSetsCrossValidation,1);

Weight_init = Weight_min+(Weight_max-Weight_min)*rand(s,1,size(x,2)); %W{layer}(neurona capa previa,neurona capa posterior)

%Definicion de datos de prueba y entrenamiento
index = crossvalind('Kfold', size(x,1), NumSetsCrossValidation);
for i = 1:NumSetsCrossValidation % toma la parte i-ésima como muestra de prueba y las otras partes como muestra de entrenamiento
   test = (index == i);% Retornar indices del fold actual para test
   train = ~test; % Todos los demas son prueba
   x_training = x(train, :);
   x_testing = x(test, :);
   y_training = y(train);
   y_testing = y(test);

%     % Particion Entranamiento / Prueba
%     index = randsample(s,1:size(x,1),round(testingDataPercent*size(x,1)/100));
%     x_testing = x(index,:);
%     y_testing = y(index);
%     x_training = x(setdiff(1:end,index),:);
%     y_training = y(setdiff(1:end,index));
    
    
    %Inicializacion
    Weight = Weight_init;%Weight_min+(Weight_max-Weight_min)*rand(s,1,size(x_training,2)); %W{layer}(neurona capa previa,neurona capa posterior)
    %Weight = 2*rand(s,1,size(x_training,2))-1; %rango entre -1 y 1
    W_bolsillo = Weight;
    
    %Training
    iteration = 0;
    Errores = []; %
    ErroresTesting = [];
    E = Emin + 1; % Para que entre en el while
    t=0;% Contador de iteraciones consecutivas sin equivocarse
    while iteration < IterationsMax && E > Emin  
        iteration = iteration + 1;
        k = randi(s,size(x_training,1)); %indice de los datos de entrada k
        %k = mod(iteration,size(x_training,1))+1;
        
        weighted_sum = Weight * x_training(k,:)';%Estado interno de la neurona
        if strcmp(tipo,'LMS')
            %LMS durante el entrenamiento saca errores con base en el
            %estado interno, no la activacion
            output = weighted_sum;%
            error = output - y_training(k);
            dWeight = mu * error * x_training(k,:);
            Weight = Weight - dWeight;
        else 
            z_k = weighted_sum*y_training(k);%Estado interno de la neurona para el ejemplo k
            if sign(z_k) < 0 % Verificamos si hay error
                % porque el cambio depende del ejemplo y no del error??
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
        
        % Error Global
        z = Weight*x_training';% Todos las salidas sin umbralizar para todos los ejemplos
        z = sign(z); %Umbral igual para LMS y perceptron por clases -1 y 1
        cantidadErrores = sum(y_training~=z');%Cantidad de errores para los pesos actuales
        E = sum((y_training - z').^2)/2; % El doble de la cantidad de errores actuales
        maxErrores = sum((2*ones(size(y_training))).^2)/2; % Cantidad maxima que se puede equivocar
        E = E / maxErrores; % Error porcentual del peso actual sobre todo el conjunto de entrenamiento
        Errores = [Errores;E];% Historial a lo largo de la iteraciones del error porcentual del conjunto de entrenamiento
                
        %Error de prueba
        z_testing = Weight * x_testing';
        z_testing = sign(z_testing);
        E_testing = sum(y_testing~=z_testing');
        E_testing = E_testing / (2*size(x_testing,1));
        ErroresTesting = [ErroresTesting; E_testing];
    end
    % Grafica del error global de entrenamiento y prueba
    % A lo largo de las iteraciones
    f=figure(i);
    plot(Errores,'b');
    xlabel('iteraciones');
    ylabel('Error Global');
    hold on;
    plot(ErroresTesting,'r');
    hold off;
    legend('de entrenamiento','de prueba')
    title2=['Error Global vs iteraciones para  fold ',num2str(i)];
    title(title2)
    saveas(i,[dt title2 '.png']) 
   
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
    
    Iterations(i) = iteration;
    ErrorTraining(i) = E;
    ErrorTesting(i) = E_testing;
end



tipo = {tipo};
% Create a table with the data and variable names
TPARAMS = table(Weight_max,Weight_min,NumSetsCrossValidation,mu,Emin,seed,IterationsMax,tipo,tmax)
% Write data to text file
writetable(TPARAMS, [dt  'params.txt'])
Fold = [1:NumSetsCrossValidation]';
table = table(Fold,Iterations,ErrorTraining,ErrorTesting)
writetable(table, [dt  'folds.txt'])



