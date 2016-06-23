//
//  MLNNeuralNet.h
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//  

#import <Foundation/Foundation.h>

@interface MLNNeuralNet : NSObject

//initializer
-(instancetype)initWithInputs:(int)inputs hiddenSize:(int)hidden;

//Helpers
-(NSMutableArray *)transpose:(NSArray *)array;
-(NSNumber *)sigmoid:(NSNumber *)number;
-(NSNumber *)derivative:(NSNumber *)number;

//Vector and Matrix Math
-(NSMutableArray *)sigmoidForVector:(NSArray *)values;
-(NSMutableArray *)sigmoidForMatrix:(NSArray *)values;
-(NSMutableArray *)sigmoidDerivativeForVector:(NSArray *)values;
-(NSMutableArray *)sigmoidDerivativeForMatrix:(NSArray *)values;
-(NSMutableArray *)addVector:(NSArray *)vector1 toVector:(NSArray *)vector2;
-(NSMutableArray *)multiplyVectorElements:(NSArray *)vector1 by:(NSArray *)vector2;
-(NSMutableArray *)dotProductMatrix:(NSArray *)matrix byVector:(NSArray *)vector;
-(NSMutableArray *)addMatrix:(NSArray *)matrix1 toMatrix:(NSArray *)matrix2;
-(NSMutableArray *)outerProduct:(NSMutableArray *)array1 by:(NSMutableArray *)array2;
-(double)vectorDotProduct:(NSArray *)vector1 by:(NSArray *)vector2;

//Neural Network Methods
-(NSMutableArray *)createLayerWithNeurons:(int)numberOfNeurons withInputs:(int)numberOfInputs;
-(void)train:(NSArray *)inputs trainingOutput:(NSArray *)expectedOutput iterations:(int)iterations;
-(void)predict:(NSArray *)testArray;


@end
