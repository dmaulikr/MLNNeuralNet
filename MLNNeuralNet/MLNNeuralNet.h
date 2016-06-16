//
//  MLNNeuralNet.h
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MLNNeuralNet : NSObject

-(void)train:(NSArray *)inputs trainingOutput:(NSArray *)expectedOutput iterations:(int)iterations;
-(void)predict:(NSArray *)testArray;
-(NSMutableArray *)transpose:(NSArray *)array;

@end
