//
//  MLNNeuralNet.m
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//
/*  MLN is a multi-layer Artificial Neural Network developed in Objective-C. MLN can be used to accept a custom number of inputs and pass through a single custom hidden layer achitecture (# of neurons) to predict non-linear patterns in data and make predictions about new patterns. MLN was inspired, in part, by Milo Harper's ANN written in Python and Andrej Karpathy's excellent posts on neural networks. Future releases will allow more customized hidden/output architectures and Recurrent Neural Network patterns.*/

#import "MLNNeuralNet.h"

@interface MLNNeuralNet ()

/*Synapses between input and hidden layer - will be initialized as a 2-D matrix (NSMutableArray of NSMutableArrays of synapse weight values*/
@property (strong, nonatomic) NSMutableArray *wxh;

/*Synapses between hidden layer and output - initialized with vector (NSMutableArray)*/
@property (strong, nonatomic) NSMutableArray *why;

@end

@implementation MLNNeuralNet

#pragma mark - Initializer

-(instancetype)initWithInputs:(int)inputs hiddenSize:(int)hidden {
    self = [super init];
    
    if (self) {
        _wxh = [self createLayerWithNeurons:hidden withInputs:inputs];
        _why = [self createLayerWithNeurons:1 withInputs:hidden];
    }
    
    return self;
}

-(instancetype)init {
    self = [super init];
    
    if (self) {
        
        _wxh = [self createLayerWithNeurons:5 withInputs:3];
        
        _why = [self createLayerWithNeurons:1 withInputs:5];
        
        /*_wxh = [[NSMutableArray alloc] initWithArray:@[@[@(-0.16595599), @(0.44064899), @(-0.99977125), @(-0.39533485)],
                                                       @[@(-0.70648822), @(-0.81532281), @(-0.62747958), @(-0.30887855)],
                                                       @[@(-0.20646505), @(0.07763347), @(-0.16161097), @(0.370439)]
                                                       ]];
        _why = [[NSMutableArray alloc] initWithArray:@[@(-0.5910955), @(0.75623487), @(-0.94522481), @(0.34093502)]];*/
    }
    
    return self;
}

#pragma mark - Neural Architecture
-(NSMutableArray *)createLayerWithNeurons:(int)numberOfNeurons withInputs:(int)numberOfInputs {
    /* 
     Neurons -> columns
     Inputs -> rows (row 1 = input 1 for all neurons, row 2 = input 2 for all neurons, etc...)
     */
    
    /*Returns a vector of random weights (NSNumber) if only 1 neuron, otherwise returns a 2-dimenionsal matrix */
    NSMutableArray *layer = [[NSMutableArray alloc] init];
    
    if (numberOfNeurons == 1) {
        for (int i = 0; i < numberOfInputs; i++) {
            [layer addObject:@([self randomWeight])];
        }
    }
    else {
        //iterate through the number of inputs in the layer
        for (int i = 0; i < numberOfInputs; i++) {
            //create the input array
            NSMutableArray *inputs = [[NSMutableArray alloc] init];
            
            //iterate through inputs and add weights for each neuron in the layer
            for (int j = 0; j < numberOfNeurons; j++) {
                [inputs addObject:@([self randomWeight])];
            }
            //add the neuron to the layer
            [layer addObject:inputs];
        }
    }
    
    return layer;
}

#pragma mark - Training

-(void)train:(NSArray *)inputs trainingOutput:(NSArray *)expectedOutput iterations:(int)iterations {
    
    //setup NSDate for tracking training times
    NSDate *startTime = [NSDate date];
    
    for (int i = 0; i < iterations; i++) {
        
        //pass the input through the network
        NSMutableArray *layer_1_out = [self processInput:inputs layer:1];
        //NSLog(@"layer 1 output: %@", layer_1_out);
        NSMutableArray *layer_2_out = [self processInput:layer_1_out layer:2];
        //NSLog(@"layer 2 output: %@", layer_2_out);
        
        //calculate the error for layer 2
        NSMutableArray *layer_2_error = [[NSMutableArray alloc] init];
        for (int i = 0; i < [layer_2_out count]; i++) {
            [layer_2_error addObject:@([[expectedOutput objectAtIndex:i] doubleValue] - [[layer_2_out objectAtIndex:i] doubleValue])];
        }
        
        //NSLog(@"layer 2 error: %@", layer_2_error);
        
        //calculate the layer 2 delta
        NSMutableArray *layer_2_delta = [self multiplyVectorElements:layer_2_error by:[self sigmoidDerivativeForVector:layer_2_out]];
        //NSLog(@"layer 2 delta: %@", layer_2_delta);
        
        //layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
        //layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
        
        NSMutableArray *layer_1_error = [self outerProduct:layer_2_delta by:self.why];
        
        //NSLog(@"layer 1 error: %@", layer_1_error);
        
        NSMutableArray *sigmoidDerivative = [self sigmoidDerivativeForMatrix:layer_1_out];
        //NSLog(@"sig deriv: %@", sigmoidDerivative);
        
        NSMutableArray *layer_1_delta = [self multiplyMatrixElements:layer_1_error by:sigmoidDerivative];
        //NSLog(@"layer 1 delta: %@", layer_1_delta);
        
        NSMutableArray *layer_1_adjustments = [self dotProduct:[self transpose:inputs] by:layer_1_delta];
        //NSLog(@"layer1 adjustment: %@", layer_1_adjustments);
        
        NSMutableArray *layer_2_adjustments = [self dotProductMatrix:[self transpose:layer_1_out] byVector:layer_2_delta];
        //NSLog(@"layer 2 adjustments: %@", layer_2_adjustments);
        
        NSMutableArray *adjusted_layer1 = [self addMatrix:self.wxh toMatrix:layer_1_adjustments];
        //NSLog(@"adjusted layer 1 synapses: %@", adjusted_layer1);
        
        NSMutableArray *adjusted_layer2 = [self addVector:self.why toVector:layer_2_adjustments];
        //NSLog(@"layer 2 adjusted: %@", adjusted_layer2);
        
        self.wxh = adjusted_layer1;
        self.why = adjusted_layer2;
        
    }
    
    //calculate how long the training took
    NSDate *endTime = [NSDate date];
    NSTimeInterval duration = [endTime timeIntervalSinceDate:startTime];
    NSLog(@"Training took: %f seconds", duration);
    
}

-(NSMutableArray *)processInput:(NSArray *)inputs layer:(int)layer {
    
    NSMutableArray *products = [[NSMutableArray alloc] init];
    
    if (layer == 1) {
        products = [self dotProduct:inputs by:self.wxh];
        return [self sigmoidForMatrix:products];
    }
    else if (layer == 2) {
        products = [self dotProductMatrix:inputs byVector:self.why];
        return [self sigmoidForVector:products];
    }
    else {
        return nil;
    }
    
}

-(void)predict:(NSArray *)testArray {
    
    NSMutableArray *layer_1_output = [[NSMutableArray alloc] init];
    
    NSMutableArray *transposedSynapses = [self transpose:self.wxh];
    //iterate through the layer 1 synaptic weights
    for (int i = 0; i < [transposedSynapses count]; i++) {
        NSMutableArray *slice = [transposedSynapses objectAtIndex:i];
        double layer_1_sum = 0.00;
        for (int j = 0; j < [testArray count]; j++) {
            layer_1_sum += [[testArray objectAtIndex:j] doubleValue] * [[slice objectAtIndex:j] doubleValue];
        }
        [layer_1_output addObject:@(layer_1_sum)];
    }
    
    NSLog(@"layer 1 output: %@", layer_1_output);
    
    NSMutableArray *sigmoid_layer_1 = [self sigmoidForVector:layer_1_output];
    NSLog(@"layer 1 sigmoid output %@", sigmoid_layer_1);
    
    
    double dotProduct = [self vectorDotProduct:sigmoid_layer_1 by:self.why];
    double sigmoid = 1.00 / (1.00 + (exp(-dotProduct)));
    NSLog(@"Answer: %f", dotProduct);
    NSLog(@"Sigmoid answer: %f", sigmoid);
    
}

#pragma mark - Helper Functions

-(NSMutableArray *)transpose:(NSArray *)array {
    
    if ([array count] <= 1 || ![[[array objectAtIndex:0] class] isSubclassOfClass:[NSArray class]]) {
        @throw [NSException exceptionWithName:@"Transpose Error"
                                       reason:@"Cannot transpose the given matrix."
                                     userInfo:nil];
    }
    
    NSMutableArray *transposed = [[NSMutableArray alloc] init];
    
    //populate the transposed array with the appropriate amount of container arrays
    for (int i = 0; i <[[array objectAtIndex:0] count]; i++) {
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        [transposed addObject:tempArray];
    }
    
    for (int i = 0; i < [array count]; i++) {
        NSMutableArray *slice = [array objectAtIndex:i];
        //add values to the array
        for (int j = 0; j < [slice count]; j++) {
            NSMutableArray *container = [transposed objectAtIndex:j];
            [container addObject:[slice objectAtIndex:j]];
            [transposed replaceObjectAtIndex:j withObject:container];
        }
    }
    
    return transposed;
}

-(double)randomWeight {
    double randomWeight = arc4random() % 256 / 256.0;
    if (arc4random_uniform(2) == 1) {
        randomWeight *= -1;
    }
    return randomWeight;
}

#pragma mark - Sigmoid

-(NSNumber *)sigmoid:(NSNumber *)number {
    return @(1.00 / (1.00 + exp((-number.doubleValue))));
}

-(NSMutableArray *)sigmoidForVector:(NSArray *)values {
    //converts all values in an n-dimensional vector to sigmoid
    
    NSMutableArray *sigmoids = [[NSMutableArray alloc] init];
    for (NSNumber *value in values) {
        [sigmoids addObject:[self sigmoid:value]];
    }
    
    return sigmoids;
}

-(NSMutableArray *)sigmoidForMatrix:(NSArray *)values {
    //converts all values in a 2-D matrix to sigmoid
    
    NSMutableArray *sigmoids = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [values count]; i++) {
        NSMutableArray *slice = [values objectAtIndex:i];
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (NSNumber *number in slice) {
            [tempArray addObject:[self sigmoid:number]];
        }
        [sigmoids addObject:tempArray];
        tempArray = nil;
    }
    
    return sigmoids;
}

#pragma mark - Derivative

-(NSNumber *)derivative:(NSNumber *)number {
    return @(number.doubleValue * (1.00 - number.doubleValue));
}

-(NSMutableArray *)sigmoidDerivativeForVector:(NSArray *)values {
    
    NSMutableArray *derivatives = [[NSMutableArray alloc] init];
    
    for (NSNumber *number in values) {
        [derivatives addObject:[self derivative:number]];
    }
    return derivatives;
}

-(NSMutableArray *)sigmoidDerivativeForMatrix:(NSArray *)values {
    
    NSMutableArray *derivatives = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [values count]; i++) {
        NSMutableArray *slice = [values objectAtIndex:i];
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (NSNumber *number in slice) {
            [tempArray addObject:[self derivative:number]];
        }
        [derivatives addObject:tempArray];
        tempArray = nil;
    }
    
    return derivatives;
}

#pragma mark - Matrix Addition

-(NSMutableArray *)addVector:(NSArray *)vector1 toVector:(NSArray *)vector2 {
    
    NSMutableArray *resultVector = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [vector1 count]; i++) {
        [resultVector addObject:@([[vector1 objectAtIndex:i] doubleValue] + [[vector2 objectAtIndex:i] doubleValue])];
    }
    
    return resultVector;
}

-(NSMutableArray *)addMatrix:(NSArray *)matrix1 toMatrix:(NSArray *)matrix2 {
    
    NSMutableArray *resultMatrix = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [matrix1 count]; i++) {
        NSMutableArray *array1_slice = [matrix1 objectAtIndex:i];
        NSMutableArray *array2_slice = [matrix2 objectAtIndex:i];
        
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (int j = 0; j < [array1_slice count]; j++) {
            [tempArray addObject:@([[array1_slice objectAtIndex:j] doubleValue] + [[array2_slice objectAtIndex:j] doubleValue])];
        }
        [resultMatrix addObject:tempArray];
        tempArray = nil;
             
    }
    
    return resultMatrix;
}

#pragma mark - Matrix Multiplication

-(NSMutableArray *)multiplyVectorElements:(NSArray *)vector1 by:(NSArray *)vector2 {
    
    /*Calculates element-wise vector multiplication.  The output is a vector*/
    
    if ([vector1 count] != [vector2 count]) {
        @throw [NSException exceptionWithName:@"Vector Size Mismatch"
                                       reason:@"Cannot multiply vector elements of different size"
                                     userInfo:nil];
    }
    
    NSMutableArray *result = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [vector1 count]; i++) {
        [result addObject:@([[vector1 objectAtIndex:i] doubleValue] * [[vector2 objectAtIndex:i] doubleValue])];
    }
    
    return result;
}
/////
/*-(NSMutableArray *)outerProduct:(NSMutableArray *)array1 by:(NSMutableArray *)array2 {
    
    NSMutableArray *result = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [array2 count]; i++) {
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (int j = 0; j < [array1 count]; j++) {
            double product = [[array2 objectAtIndex:i] doubleValue] * [[array1 objectAtIndex:j] doubleValue];
            NSLog(@"%f * %f", [[array2 objectAtIndex:i] doubleValue], [[array1 objectAtIndex:j] doubleValue]);
            [tempArray addObject:@(product)];
        }
        [result addObject:tempArray];
    }
    
    return [self transpose:result];
}*/

-(NSMutableArray *)outerProduct:(NSMutableArray *)matrix1 by:(NSMutableArray *)matrix2 {
    
    /*Example: if matrix1 is @[2, 4, 6] and matrix2 @[3, 4, 5], then calculation is:
     [2 * 3, 2 * 4, 2 * 5], [4 * 3, etc...]
     and result is:
     @[@[6, 8, 10], @[12, 16, 20], @[18, 24, 30]
     */
    
    NSMutableArray *result = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [matrix1 count]; i++) {
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (int j = 0; j < [matrix2 count]; j++) {
            double product = [[matrix1 objectAtIndex:i] doubleValue] * [[matrix2 objectAtIndex:j] doubleValue];
            [tempArray addObject:@(product)];
        }
        [result addObject:tempArray];
    }
    
    return result;
}

/////
-(NSMutableArray *)dotProductMatrix:(NSArray *)matrix byVector:(NSArray *)vector {
    
    /*Calculates matrix-vector product*/
    
    NSMutableArray *sumArray = [[NSMutableArray alloc] init];
    for (int i = 0; i < [matrix count]; i++) {
        NSMutableArray *slice = [matrix objectAtIndex:i];
        
        if ([slice count] != [vector count]) {
            @throw [NSException exceptionWithName:@"Matrix-Vector Size Mismatch"
                                           reason:@"Cannot calculate dot product of matrix and vector of different sizes."
                                         userInfo:nil];
        }
        
        double sum = 0.00;
        for (int j = 0; j < [slice count]; j++) {
            sum += [[slice objectAtIndex:j] doubleValue] * [[vector objectAtIndex:j] doubleValue];
        }
        [sumArray addObject:@(sum)];
    }

    return sumArray;
}
/////
-(NSMutableArray *)dotProduct:(NSArray *)array1 by:(NSArray *)array2 {
    
    /*Calculates dot product for 2 matrices*/
    
    NSMutableArray *transposed = [self transpose:array2];
    NSMutableArray *resultArray = [[NSMutableArray alloc] init];
        
        for (int i = 0; i < [array1 count]; i++) {
            NSMutableArray *array1_slice = [array1 objectAtIndex:i];
            NSMutableArray *tempArray = [[NSMutableArray alloc] init];
            
            //iterate through the second array
            for (int j = 0; j < [transposed count]; j++) {
                
                NSMutableArray *array2_slice = [transposed objectAtIndex:j];
                double sum = 0.00;
                
                for (int k = 0; k < [array1_slice count]; k++) {
                    sum += [[array1_slice objectAtIndex:k] doubleValue] * [[array2_slice objectAtIndex:k] doubleValue];
                }
                [tempArray addObject:@(sum)];
            }
            [resultArray addObject:tempArray];
            tempArray = nil;
        }

        return resultArray;
}

-(double)vectorDotProduct:(NSArray *)vector1 by:(NSArray *)vector2 {
    
    /*Calculates dot product of 2 vectors*/
    
    if ([vector1 count] != [vector2 count]) {
        @throw [NSException exceptionWithName:@"Vector Size Mismatch"
                                       reason:@"Cannot calculate dot product between vectors of different sizes"
                                     userInfo:nil];
    }
    
    double returnValue = 0.00;
    for (int i = 0; i < [vector1 count]; i++) {
        returnValue += [[vector1 objectAtIndex:i] doubleValue] * [[vector2 objectAtIndex:i] doubleValue];
    }
    return returnValue;
}
/////
-(NSMutableArray *)multiplyMatrixElements:(NSArray *)array1 by:(NSArray *)array2 {
    
    /*Element-wise matrix multiplication. Output is a matrix*/
    
    NSMutableArray *resultArray = [[NSMutableArray alloc] init];
    for (int i = 0; i < [array1 count]; i++) {
        NSMutableArray *slice1 = [array1 objectAtIndex:i];
        NSMutableArray *slice2 = [array2 objectAtIndex:i];
        
        if ([slice1 count] != [slice2 count]) {
            @throw [NSException exceptionWithName:@"Matrix Size Mismatch"
                                           reason:@"Cannot multiply matrix elements of different size"
                                         userInfo:nil];
        }
        
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (int j = 0; j < [slice1 count]; j++) {
            double product = [[slice1 objectAtIndex:j] doubleValue] * [[slice2 objectAtIndex:j] doubleValue];
            [tempArray addObject:@(product)];
        }
        
        [resultArray addObject:tempArray];
        tempArray = nil;
    }
    return resultArray;
}

/*
-(NSMutableArray *)dotProduct:(NSArray *)array1 by:(NSArray *)array2 {
    
    NSMutableArray *products = [[NSMutableArray alloc] init];
    for (int i = 0; i < [array1 count]; i++) {
        [products addObject:@(1)];
    }
    
    for (int i = 0; i < [array1 count]; i++) {
        NSArray *slice = [array1 objectAtIndex:i];
        double product = 1.00;
        for (int j = 0; j < [slice count]; j++) {
            NSNumber *number = @([[slice objectAtIndex:j] doubleValue] * [[array2 objectAtIndex:j] doubleValue]);
            product *= ([[products objectAtIndex:i] doubleValue] * number.doubleValue);
        }
        [products replaceObjectAtIndex:i withObject:@(product)];
    }
    return products;
}*/

@end

