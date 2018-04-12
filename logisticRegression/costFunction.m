function [costJ, gradient] = costFunction(theta, X, y)

numberOfTrainingExamples = length(y);
  
costJ = 0; 
gradient = zeros(size(theta));

hypothesis = sigmoid(X*theta);
 
costJ = (-1/numberOfTrainingExamples) * sum( y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis) );


for i = 1:numberOfTrainingExamples
	gradient = gradient + ( hypothesis(i) - y(i) ) * X(i, :)';
end

gradient = (1/numberOfTrainingExamples) * gradient;

end
