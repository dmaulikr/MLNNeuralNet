//
//  MLNNeuralNet.h
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//  

#import <Foundation/Foundation.h>

@interface MLNNeuralNet : NSObject

//Initializer
-(instancetype)initWithInputs:(int)inputs hiddenSize:(int)hidden;

//Convenience initializer
+(instancetype)neuralNetWithInputs:(int)inputs;

//Accessing neural net layers readonly in public interface
@property (readonly) NSMutableArray *wxh;
@property (readonly) NSMutableArray *why;

//Helpers
-(NSMutableArray *)transpose:(NSArray *)array;
-(NSNumber *)sigmoid:(NSNumber *)number;
-(NSNumber *)derivative:(NSNumber *)number;

//Saving and Loading Synaptic Weights
-(void)saveSynapticWeights;
-(void)loadSynapticWeights;

//Vector and Matrix Math
-(NSMutableArray *)sigmoidForVector:(NSArray *)values;
-(NSMutableArray *)sigmoidForMatrix:(NSArray *)values;
-(NSMutableArray *)derivativeForVector:(NSArray *)values;
-(NSMutableArray *)derivativeForMatrix:(NSArray *)values;
-(NSMutableArray *)addVector:(NSArray *)vector1 toVector:(NSArray *)vector2;
-(NSMutableArray *)addMatrix:(NSArray *)matrix1 toMatrix:(NSArray *)matrix2;
-(NSMutableArray *)multiplyVectorElements:(NSArray *)vector1 by:(NSArray *)vector2;
-(NSMutableArray *)multiplyMatrixElements:(NSArray *)array1 by:(NSArray *)array2;
-(double)vectorDotProduct:(NSArray *)vector1 by:(NSArray *)vector2;
-(NSMutableArray *)dotProductMatrix:(NSArray *)matrix byVector:(NSArray *)vector;
-(NSMutableArray *)dotProduct:(NSArray *)array1 by:(NSArray *)array2;
-(NSMutableArray *)outerProduct:(NSMutableArray *)array1 by:(NSMutableArray *)array2;

//Neural Network Methods
-(void)train:(NSArray *)inputs trainingOutput:(NSArray *)expectedOutput iterations:(int)iterations;
-(void)predict:(NSArray *)testArray;


@end
