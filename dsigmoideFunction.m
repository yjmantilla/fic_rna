function y = dsigmoideFunction(x)
    y=sigmoideFunction(x).*(1.-sigmoideFunction(x));
end