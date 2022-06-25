rng(3);

X  = zeros(5, 5, 10);
 
X(:, :, 1) = [ 0 1 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 1 1 1 0
             ];
 
X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 0;
               1 1 1 1 1
             ];
 
X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];

X(:, :, 4) = [ 0 0 0 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               1 1 1 1 1;
               0 0 0 1 0
             ];
         
X(:, :, 5) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];

X(:, :, 6) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 1;
               1 0 0 0 1;
               1 1 1 1 1
             ];

X(:, :, 7) = [ 1 1 1 1 1;
               0 0 0 1 0;
               0 0 1 0 0;
               0 1 0 0 0;
               1 0 0 0 0
             ];

X(:, :, 8) = [ 0 1 1 1 0;
               1 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               0 1 1 1 0
             ];

X(:, :, 9) = [ 0 1 1 1 0;
               1 0 0 1 0;
               0 1 1 1 0;
               0 0 0 1 0;
               0 0 0 1 0
             ];

X(:, :, 10) = [ 0 0 1 0 0;    %For Zero
                0 1 0 1 0;
                1 0 0 0 1;
                0 1 0 1 0;
                0 0 1 0 0
             ];

D = [ 1 0 0 0 0 0 0 0 0 0;
      0 1 0 0 0 0 0 0 0 0;
      0 0 1 0 0 0 0 0 0 0;
      0 0 0 1 0 0 0 0 0 0;
      0 0 0 0 1 0 0 0 0 0;
      0 0 0 0 0 1 0 0 0 0;
      0 0 0 0 0 0 1 0 0 0;
      0 0 0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 0 0 1 0;
      0 0 0 0 0 0 0 0 0 1;
    ];
      
W1 = 2*rand(50, 25) - 1;
W2 = 2*rand( 10, 50) - 1;

for epoch = 1:10000           % train
  [W1 W2] = MultiClass(W1, W2, X, D);
end

X(:, :, 1) = [ 0 0 1 1 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 1 1 1 0
             ];
 
X(:, :, 2) = [ 0 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 1
             ];
 
X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 1 0;
               0 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];

X(:, :, 4) = [ 0 0 0 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               0 1 1 1 1;
               0 0 0 1 1
             ];
         
X(:, :, 5) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 0;
               0 0 0 1 0;
               1 1 1 1 0
             ];

X(:, :, 6) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 1;
               0 1 0 0 1;
               1 1 1 1 1
             ];

X(:, :, 7) = [ 0 1 1 1 1;
               0 0 0 1 0;
               0 0 1 0 0;
               0 1 1 0 0;
               1 0 0 0 0
             ];

X(:, :, 8) = [ 0 1 1 1 0;
               1 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               0 1 1 1 0
             ];

X(:, :, 9) = [ 0 1 1 1 0;
               1 0 0 1 0;
               1 1 1 1 0;
               0 0 0 1 0;
               0 0 0 0 0
             ];

X(:, :, 10) = [ 0 0 1 0 0;    %For Zero
                0 1 0 1 0;
                1 0 0 1 0;
                0 1 0 1 0;
                0 0 1 0 0
             ];

D = [ 1 0 0 0 0 0 0 0 0 0;
      0 1 0 0 0 0 0 0 0 0;
      0 0 1 0 0 0 0 0 0 0;
      0 0 0 1 0 0 0 0 0 0;
      0 0 0 0 1 0 0 0 0 0;
      0 0 0 0 0 1 0 0 0 0;
      0 0 0 0 0 0 1 0 0 0;
      0 0 0 0 0 0 0 1 0 0;
      0 0 0 0 0 0 0 0 1 0;
      0 0 0 0 0 0 0 0 0 1;
    ];
      

Z  = zeros(10, 10);
correct=0;                     %To check digits identified correctly
N = 10;                        % inference
for k = 1:N
  x  = reshape(X(:, :, k), 25, 1);
  v1 = W1*x;
  y1 = ReLU(v1);
  v  = W2*y1;
  y  = Softmax(v);
  Z(k, :)=y;
  correct = correct+y(k);
end

Z
accuracy = (correct/10)*100;
fprintf("\t Accuracy is %d percent \n", accuracy);