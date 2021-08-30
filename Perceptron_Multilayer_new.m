function [Table,table2,foldWeights,foldBias] = Perceptron_Multilayer_new(x,y,NeuronsPerLayer,mu,Emin,IterationsMax,NumSetsCrossValidation,testingDataPercent,Weight_min,Weight_max,seed)

dt = datestr(now,'yyyy_mmmm_dd_HH_MM_SS_FFF');
%Parametros
OUTPUT_FOLDER = '.multioutput';
create_if(OUTPUT_FOLDER);

save([OUTPUT_FOLDER '/' dt '_params.mat'],'NeuronsPerLayer','mu','Emin','IterationsMax','NumSetsCrossValidation','testingDataPercent','Weight_min','Weight_max','seed');

L = size(NeuronsPerLayer,2);

%assert(size(NeuronsPerLayer,2) == L);
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
rng(seed, 'twister');
index = crossvalind('Kfold', size(x,1), NumSetsCrossValidation);
weights_bolsillo = {};
bias_bolsillo = {};
printEachNiterations=100;
NeuronsPerLayer = [size(x,2),NeuronsPerLayer,1]; %Agrega capa inicial de datos de entrada y capa final de salida
L = L+2; %Capas ocultas de entrada más 2 de datos iniciales y salida
foldWeights={};
foldBias={};
figure
for j = 1:NumSetsCrossValidation % toma la parte i-ésima como muestra de prueba y las otras 3 partes como muestra de entrenamiento
    disp(['Fold ' num2str(j) ' of ' num2str(NumSetsCrossValidation)])
    if (NumSetsCrossValidation > 1)
   test = (index == j);% Retornar indices del fold actual para test
   train = ~test; % Todos los demas son prueba
   x_training = x(train, :);
   x_testing = x(test, :);
   y_training = y(train);
   y_testing = y(test);
   else
    % Particion Entranamiento / Prueba
    index = randsample(s,1:size(x,1),round(testingDataPercent*size(x,1)/100));
    x_testing = x(index,:);
    y_testing = y(index);
    x_training = x(setdiff(1:end,index),:);
    y_training = y(setdiff(1:end,index));
   end    
    %Inicializacion
    weights = {[]};
    bias = {[]};
    for l = 1:L-1
        weights{end+1} = Weight_min+(Weight_max-Weight_min)*rand(s,NeuronsPerLayer(l),NeuronsPerLayer(l+1)); %W{layer}(neurona capa previa,neurona capa posterior) 
        bias{end+1} = rand(s,1,NeuronsPerLayer(l+1)); %bias por capa
    end
    
    iterations = 0;
    E=Emin+1;
    Errores_training_fold=[0];
    Errores_testing_fold=[];
    while E>Emin && iterations < IterationsMax
        if mod(iterations,printEachNiterations)== 0
            fprintf('%d ', iterations); 
        end
        iterations = iterations + 1;
        %fprintf([num2str(iterations) ' ']);
        %k = randi(size(x_training,1)); %indice de los datos de entrada k
        k = mod(iterations,size(x_training,1))+1;
        
        [a,z] = Feedforward(L,weights,x_training(k,:),bias);
        
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
        
        ErrorGlobalTraining = getGlobalError(L,x_training,y_training,weights,bias);
        if ErrorGlobalTraining < Errores_training_fold(end)
            weights_bolsillo = weights;
            bias_bolsillo = bias;
        end
        Errores_training_fold = [Errores_training_fold; ErrorGlobalTraining];
        Errores_testing_fold = [Errores_testing_fold; getGlobalError(L,x_testing,y_testing,weights,bias)];
        E = ErrorGlobalTraining;
    end
    
    subplot(ceil(sqrt(NumSetsCrossValidation)),floor(sqrt(NumSetsCrossValidation)),j)
    plot(Errores_training_fold);
%     hold on
%     plot(Errores_testing_fold);
%     xlabel('iteraciones');
%     ylabel('Error Global');
%     legend('de entrenamiento','de prueba')
%     title2=['Error Global vs iteraciones para  fold ',num2str(j)];
%     title(title2)
%     hold off
    
    if Errores_training_fold(end) > Emin 
        disp('Uso el bolsillo');
        ErrorTraining(j) = getGlobalError(L,x_training,y_training,weights_bolsillo,bias_bolsillo);
        ErrorTesting(j) = getGlobalError(L,x_testing,y_testing,weights_bolsillo,bias_bolsillo);
        Weights = weights_bolsillo;
        Bias = bias_bolsillo;
    else
        ErrorTraining(j) = Errores_training_fold(end);
        ErrorTesting(j) = Errores_testing_fold(end);
        Weights = weights;
        Bias=bias;
    end
    foldWeights{end+1} = Weights;
    foldBias{end+1}=Bias;
    Iterations(j) = iterations;
end
Fold = [1:NumSetsCrossValidation]';
Table = table(Fold,Iterations,ErrorTraining,ErrorTesting);
saveas(1,[OUTPUT_FOLDER '/' dt 'ErrorGlobalvsIteracionesMultilayer.png'])
writetable(Table, [OUTPUT_FOLDER '/' dt  '_folds.txt'])

MeanTraining=round(mean(ErrorTraining),4);
MeanTesting=round(mean(ErrorTesting),4);
STDTraining=round(std(ErrorTraining),4);
STDTesting=round(std(ErrorTesting),4);


table2 = table(MeanTraining,MeanTesting,STDTraining,STDTesting);
writetable(table2, [OUTPUT_FOLDER '/' dt  '_crossval.txt'])
end

function [a,z] = Feedforward(L,weights,x,bias)
%Feedforward
a = {x};
z = {[]};
    for l = 2:L
        z{end+1} = weights{l}'*a{l-1}'+bias{l}';
        if l == L
            a{end+1} = z{end};
        else
            a{end+1} = sigmoideFunction(z{end});
        end
    end
end

function E = getGlobalError(L,x,y,weights,bias)
output=zeros(size(y));
%Calculo de la salida:
    for i=1:size(x,1)
        [a,~] = Feedforward(L,weights,x(i,:),bias);
        output(i) = a{end};
    end
    output = sign(output);
    E = sum((y - output).^2)/2;
    maxErrores = sum((2*ones(size(y))).^2)/2;
    E = E / maxErrores;
end

function create_if(yourFolder)
    if not(isfolder(yourFolder))
    mkdir(yourFolder)
    end
end

