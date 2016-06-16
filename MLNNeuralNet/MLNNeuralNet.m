//
//  MLNNeuralNet.m
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//

#import "MLNNeuralNet.h"

@interface MLNNeuralNet ()

//the synaptic weights have been changed to a vector to a matrix
@property (strong, nonatomic) NSMutableArray *wxh;
@property (strong, nonatomic) NSMutableArray *why;

@end

@implementation MLNNeuralNet

-(instancetype)init {
    self = [super init];
    
    if (self) {
        
        _wxh = [self createNeuronLayer:4 withInputs:3];
        
        _why = [self createNeuronLayer:1 withInputs:4];
        
        for (int i = 0; i < 2000; i++) {
            NSLog(@"%f", [self randomWeight]);
        }
        
        /*_wxh = [[NSMutableArray alloc] initWithArray:@[@[@(-0.16595599), @(0.44064899), @(-0.99977125), @(-0.39533485)],
                                                       @[@(-0.70648822), @(-0.81532281), @(-0.62747958), @(-0.30887855)],
                                                       @[@(-0.20646505), @(0.07763347), @(-0.16161097), @(0.370439)]
                                                       ]];
        _why = [[NSMutableArray alloc] initWithArray:@[@(-0.5910955), @(0.75623487), @(-0.94522481), @(0.34093502)]];*/
    }
    
    return self;
}

#pragma mark - Creations
-(NSMutableArray *)createNeuronLayer:(int)numberOfNeurons withInputs:(int)numberOfInputs {
    /* 
     Neurons -> columns
     Inputs -> rows (row 1 = inputs 1 for all neurons, row 2 = inputs 2 for all neurons, etc...)
     */
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
            //[layer addObject:inputs];
            
            //iterate through inputs and add connection for each neuron in the layer
            for (int j = 0; j < numberOfNeurons; j++) {
                [inputs addObject:@([self randomWeight])];
            }
            //add the neuron to the layer
            [layer addObject:inputs];
        }
    }
    
    return layer;
}

#pragma mark - Math and Helper Functions

-(double)randomWeight {
    double randomWeight = arc4random() % 256 / 256.0;
    if (arc4random_uniform(2) == 1) {
        randomWeight *= -1;
    }
    return randomWeight;
}

-(NSMutableArray *)sigmoid:(NSArray *)values {
    
    
    NSMutableArray *sigmoids = [[NSMutableArray alloc] init];
    
    for (NSNumber *value in values) {
        double sigmoid = 1.00 / (1.00 + (exp(-value.doubleValue)));
        [sigmoids addObject:@(sigmoid)];
    }
    
    return sigmoids;
}

-(NSMutableArray *)sigmoidForMatrix:(NSArray *)inputs {
    
    
    
    NSMutableArray *sigmoids = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [inputs count]; i++) {
        NSMutableArray *slice = [inputs objectAtIndex:i];
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (NSNumber *number in slice) {
            double sigmoid = 1.00 / (1.00 + (exp(-number.doubleValue)));
            [tempArray addObject:@(sigmoid)];
        }
        [sigmoids addObject:tempArray];
        tempArray = nil;
    }
    
    return sigmoids;
}


-(NSMutableArray *)sigmoidDerivative:(NSArray *)values {
    
    NSMutableArray *derivatives = [[NSMutableArray alloc] init];
    for (NSNumber *number in values) {
        double derivative = [number doubleValue] * (1.00 - [number doubleValue]);
        [derivatives addObject:@(derivative)];
    }
    
    return derivatives;
}

-(NSMutableArray *)sigmoidDerivativeForMatrix:(NSArray *)values {
    
    NSMutableArray *derivatives = [[NSMutableArray alloc] init];
    //NSLog(@"preparing for sigmoid");
    
    for (int i = 0; i < [values count]; i++) {
        NSMutableArray *slice = [values objectAtIndex:i];
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (NSNumber *number in slice) {
            double derivative = [number doubleValue] * (1.00 - [number doubleValue]);
            [tempArray addObject:@(derivative)];
        }
        [derivatives addObject:tempArray];
        tempArray = nil;
    }
    
    return derivatives;
}

-(NSMutableArray *)transpose:(NSArray *)array {
    
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

-(NSMutableArray *)addMatrix:(NSArray *)matrix1 byMatrix:(NSArray *)matrix2 {
    
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

-(NSMutableArray *)addVector:(NSArray *)vector1 toVector:(NSArray *)vector2 {
    
    NSMutableArray *resultVector = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [vector1 count]; i++) {
        [resultVector addObject:@([[vector1 objectAtIndex:i] doubleValue] + [[vector2 objectAtIndex:i] doubleValue])];
    }
    
    return resultVector;
    
}

-(NSMutableArray *)vectorMultiply:(NSArray *)array1 by:(NSArray *)array2 {
    
    if ([array1 count] != [array2 count]) {
        //NSLog(@"vectors not the same size - cannot multiply");
        return nil;
    }
    
    NSMutableArray *result = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [array1 count]; i++) {
        [result addObject:@([[array1 objectAtIndex:i] doubleValue] * [[array2 objectAtIndex:i] doubleValue])];
    }
    
    return result;
}

-(NSMutableArray *)outerProduct:(NSMutableArray *)array1 by:(NSMutableArray *)array2 {
    
    NSMutableArray *result = [[NSMutableArray alloc] init];
    
    for (int i = 0; i < [array2 count]; i++) {
        NSMutableArray *tempArray = [[NSMutableArray alloc] init];
        for (int j = 0; j < [array1 count]; j++) {
            double product = [[array2 objectAtIndex:i] doubleValue] * [[array1 objectAtIndex:j] doubleValue];
            [tempArray addObject:@(product)];
        }
        [result addObject:tempArray];
    }
    
    return [self transpose:result];
}

-(NSMutableArray *)dotProduct2D:(NSArray *)array1 by1D:(NSArray *)array2 {
    
    NSMutableArray *sumArray = [[NSMutableArray alloc] init];
    for (int i = 0; i < [array1 count]; i++) {
        NSMutableArray *slice = [array1 objectAtIndex:i];
        double sum = 0.00;
        for (int j = 0; j < [slice count]; j++) {
            //NSLog(@"slice value: %f", [[slice objectAtIndex:j] doubleValue]);
            //NSLog(@"array2 value: %f", [[array2 objectAtIndex:j] doubleValue]);
            sum += [[slice objectAtIndex:j] doubleValue] * [[array2 objectAtIndex:j] doubleValue];
        }
        //NSLog(@"adding sum to dot return array: %f", sum);
        [sumArray addObject:@(sum)];
        
    }
    //NSLog(@"sum array: %@", sumArray);
    //NSLog(@"----------------");
    return sumArray;
}

-(NSMutableArray *)dotProductMatrix:(NSArray *)array1 byVector:(NSArray *)vector {
    
    NSMutableArray *transposed = [self transpose:array1];
    NSMutableArray *sumArray = [[NSMutableArray alloc] init];
    for (int i = 0; i < [transposed count]; i++) {
        NSMutableArray *slice = [transposed objectAtIndex:i];
        double sum = 0.00;
        for (int j = 0; j < [vector count]; j++) {
            //NSLog(@"slice value: %f", [[slice objectAtIndex:j] doubleValue]);
            //NSLog(@"array2 value: %f", [[array2 objectAtIndex:j] doubleValue]);
            sum += [[slice objectAtIndex:j] doubleValue] * [[vector objectAtIndex:j] doubleValue];
        }
        //NSLog(@"adding sum to dot return array: %f", sum);
        [sumArray addObject:@(sum)];
        
    }
    //NSLog(@"sum array: %@", sumArray);
    //NSLog(@"----------------");
    return sumArray;
}

-(NSMutableArray *)dotProduct2D:(NSArray *)array1 by:(NSArray *)array2 {
    
    //TRUE MATRIX DOT PRODUCT METHOD
    //NSLog(@"inputs: %lu", [[array1 objectAtIndex:0] count]);
    //NSLog(@"%lu", [array2 count]);
    
    NSMutableArray *transposed = [self transpose:array2];
    NSLog(@"transposed: %@", transposed);
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
                    //NSLog(@"adding objects to sum: %f, %f", [[array1_slice objectAtIndex:k] doubleValue], [[array2_slice objectAtIndex:k] doubleValue]);
                }
                //NSLog(@"----------------sum: %f", sum);
                [tempArray addObject:@(sum)];
            }
            
            [resultArray addObject:tempArray];
            tempArray = nil;
            
            
        }

        return resultArray;
    

}

-(NSMutableArray *)specialDotProduct2D:(NSArray *)array1 by:(NSArray *)array2 {
    
    //TRUE MATRIX DOT PRODUCT METHOD
    //NSLog(@"inputs: %lu", [[array1 objectAtIndex:0] count]);
    //NSLog(@"%lu", [array2 count]);
    
    //NSMutableArray *transposed = [self transpose:array2];
    //NSLog(@"transposed: %@", transposed);
    NSMutableArray *resultArray = [[NSMutableArray alloc] init];
    for (int i = 0; i < [array1 count]; i++) {
        NSMutableArray *slice1 = [array1 objectAtIndex:i];
        NSMutableArray *slice2 = [array2 objectAtIndex:i];
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
}

-(double)vectorDotProduct:(NSArray *)vector1 by:(NSArray *)vector2 {
    
    double returnValue = 0.00;
    for (int i = 0; i < [vector1 count]; i++) {
        returnValue += [[vector1 objectAtIndex:i] doubleValue] * [[vector2 objectAtIndex:i] doubleValue];
    }
    return returnValue;
}



#pragma mark - Training

-(void)train:(NSArray *)inputs trainingOutput:(NSArray *)expectedOutput iterations:(int)iterations {
    
    for (int i = 0; i < iterations; i++) {
        
        //pass the input through the network
        NSMutableArray *layer_1_out = [self processInput:inputs layer:1];
        NSLog(@"layer 1 output: %@", layer_1_out);
        NSMutableArray *layer_2_out = [self processInput:layer_1_out layer:2];
        NSLog(@"layer 2 output: %@", layer_2_out);
        
        //calculate the error for layer 2
        NSMutableArray *layer_2_error = [[NSMutableArray alloc] init];
        for (int i = 0; i < [layer_2_out count]; i++) {
            [layer_2_error addObject:@([[expectedOutput objectAtIndex:i] doubleValue] - [[layer_2_out objectAtIndex:i] doubleValue])];
        }
        
        NSLog(@"layer 2 error: %@", layer_2_error);
        
        //calculate the layer 2 delta
        NSMutableArray *layer_2_delta = [self vectorMultiply:layer_2_error by:[self sigmoidDerivative:layer_2_out]];
        NSLog(@"layer 2 delta: %@", layer_2_delta);
        
        //layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
        //layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
        
        NSMutableArray *layer_1_error = [self outerProduct:layer_2_delta by:self.why];
        
        NSLog(@"layer 1 error: %@", layer_1_error);
        
        NSMutableArray *sigmoidDerivative = [self sigmoidDerivativeForMatrix:layer_1_out];
        NSLog(@"sig deriv: %@", sigmoidDerivative);
        
        NSMutableArray *layer_1_delta = [self specialDotProduct2D:layer_1_error by:sigmoidDerivative];
        NSLog(@"layer 1 delta: %@", layer_1_delta);
        
        NSMutableArray *layer_1_adjustments = [self dotProduct2D:[self transpose:inputs] by:layer_1_delta];
        NSLog(@"layer1 adjustment: %@", layer_1_adjustments);
        
        NSMutableArray *layer_2_adjustments = [self dotProduct2D:[self transpose:layer_1_out] by1D:layer_2_delta];
        NSLog(@"layer 2 adjustments: %@", layer_2_adjustments);
        
        NSMutableArray *adjusted_layer1 = [self addMatrix:self.wxh byMatrix:layer_1_adjustments];
        NSLog(@"adjusted layer 1 synapses: %@", adjusted_layer1);
        
        NSMutableArray *adjusted_layer2 = [self addVector:self.why toVector:layer_2_adjustments];
        NSLog(@"layer 2 adjusted: %@", adjusted_layer2);
        
        self.wxh = adjusted_layer1;
        self.why = adjusted_layer2;
        
        NSLog(@"layer 1 after adjustment: %@", self.wxh);
        NSLog(@"layer 2 after adjustment: %@", self.why);
        
        /*
        NSLog(@"error: %@", error);
        
        //calculate the derivatives and multiply by the error
        NSMutableArray *derivatives = [self sigmoidDerivative:output];
        NSLog(@"derivatives of output: %@", derivatives);
        
        NSMutableArray *derivativeProducts = [[NSMutableArray alloc] init];
        for (int i = 0; i < [error count]; i++) {
            NSNumber *firstNum = [error objectAtIndex:i];
            NSNumber *secondNum = [derivatives objectAtIndex:i];
            double product = [firstNum doubleValue] * [secondNum doubleValue];
            [derivativeProducts addObject:@(product)];
        }
        
        NSLog(@"derivative products: %@", derivativeProducts);
        
        //calculate how much to adjust the synaptic weights
        NSMutableArray *transposedInputs = [self transpose:inputs];//transpose inputs
        
        //NSLog(@"transposed inputs: %@", transposedInputs);
        
        NSMutableArray *adjustments = [self dotProduct2D:transposedInputs by1D:derivativeProducts];
        
        //adjust the synaptic weights
        NSLog(@"adjusting weights by: %@", adjustments);
        for (int i = 0; i < [adjustments count]; i++) {
            [self.synapticWeights replaceObjectAtIndex:i withObject:@([[self.synapticWeights objectAtIndex:i] doubleValue] + [[adjustments objectAtIndex:i] doubleValue])];
         */
        }
    
}

-(NSMutableArray *)processInput:(NSArray *)inputs layer:(int)layer {

    NSMutableArray *products = [[NSMutableArray alloc] init];
    
    if (layer == 1) {
        //NSLog(@"synaptic weights layer 1: %@", self.wxh);
        //NSLog(@"inputs: %@", inputs);
        products = [self dotProduct2D:inputs by:self.wxh];
        NSLog(@"products for input: %@", products);
        
        return [self sigmoidForMatrix:products];
    }
    else if (layer == 2) {
        //NSLog(@"synaptic weights layer 2: %@", self.why);
        //NSLog(@"inputs: %@", inputs);
        products = [self dotProduct2D:inputs by1D:self.why];
        //NSLog(@"products for input: %@", products);
        //NSLog(@"sigmoid: %@", [self sigmoid:products]);
        return [self sigmoid:products];
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
    
    NSMutableArray *sigmoid_layer_1 = [self sigmoid:layer_1_output];
    NSLog(@"layer 1 sigmoid output %@", sigmoid_layer_1);
    
    
    double dotProduct = [self vectorDotProduct:sigmoid_layer_1 by:self.why];
    double sigmoid = 1.00 / (1.00 + (exp(-dotProduct)));
    NSLog(@"Answer: %f", dotProduct);
    NSLog(@"Sigmoid answer: %f", sigmoid);
    
    /*
    NSMutableArray *layer_1_output = [self dotProductMatrix:self.wxh byVector:testArray];
    NSLog(@"predicted layer 1 output: %@", layer_1_output);
    NSMutableArray *sigmoid_layer_1 = [self sigmoid:layer_1_output];
    NSLog(@"sigmoid predict layer 1: %@", sigmoid_layer_1);
    
    double dotProduct = [self vectorDotProduct:sigmoid_layer_1 by:self.why];
    double sigmoid = 1.00 / (1.00 + (exp(-dotProduct)));
    NSLog(@"Answer: %f", dotProduct);
    NSLog(@"Sigmoid answer: %f", sigmoid);*/
    
    
                                     

}

@end

