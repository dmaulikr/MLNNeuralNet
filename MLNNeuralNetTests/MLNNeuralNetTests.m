//
//  MLNNeuralNetTests.m
//  MLNNeuralNetTests
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//

#import <XCTest/XCTest.h>
#import "MLNNeuralNet.h"

@interface MLNNeuralNet (test)

-(NSMutableArray *)dotProduct2D:(NSArray *)array1 by:(NSArray *)array2;
-(NSMutableArray *)transpose:(NSArray *)array;

@end

@interface MLNNeuralNetTests : XCTestCase

@end

@implementation MLNNeuralNetTests

- (void)setUp {
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

-(void)testMatrixDotProduct {
    
    NSMutableArray *array1 = [[NSMutableArray alloc] initWithArray:@[@[@(1), @(2), @(3)],
                        @[@(4), @(5), @(6)]
                        ]];
    NSMutableArray *correctArray = [[NSMutableArray alloc] initWithArray:@[@[@(1), @(4)]
                                                                           
                                                                           
                                                                           ]];
    
    //NSMutableArray *testArray = [self]
    
    
}

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
