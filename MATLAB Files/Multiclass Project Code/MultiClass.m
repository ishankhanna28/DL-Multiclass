function [W1, W2] = MultiClass(W1, W2, X, D)
  alpha = 0.1;
  
  N = 10;  
  for k = 1:N
    x = reshape(X(:, :, k), 25, 1);
    d = D(k, :)';
   
    v1 = W1*x;
    y1 = ReLU(v1);
    v  = W2*y1;
    y  = Softmax(v);
    
    e     = d - y;
    delta = e;

    e1     = W2'*delta;
    %Sigmoid
    %delta1 = y1.*(1-y1).*e1; 
    
    %ReLU
    delta1 = (v1 > 0).*e1;
    
    %Tanh
    %delta1 = (1-y1.*y1).*e1; 

    %Linear
    %delta1 = e1;
 
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';   
    W2 = W2 + dW2;
  end
end