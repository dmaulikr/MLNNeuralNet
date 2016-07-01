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

-(double)randomWeight;
-(NSMutableArray *)createLayerWithNeurons:(int)numberOfNeurons withInputs:(int)numberOfInputs;

@end

@interface MLNNeuralNetTests : XCTestCase

@property (strong, nonatomic) MLNNeuralNet *neuralNet;

@end

@implementation MLNNeuralNetTests

- (void)setUp {
    [super setUp];
    _neuralNet = [[MLNNeuralNet alloc] initWithInputs:3 hiddenSize:4];
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

-(void)testLayerArchitecture {
    
    NSMutableArray *correctLayer1 = [[NSMutableArray alloc] initWithArray:@[@(1), @(1)]];
    NSMutableArray *layer2Slice = [[NSMutableArray alloc] initWithArray:@[@(1), @(1)]];
    NSMutableArray *correctLayer2 = [[NSMutableArray alloc] initWithArray:@[layer2Slice, layer2Slice]];
    
    NSMutableArray *testedLayer1 = [self.neuralNet createLayerWithNeurons:1 withInputs:2];
    NSMutableArray *testedLayer2 = [self.neuralNet createLayerWithNeurons:2 withInputs:2];
    
    //vector and matrix size tests
    XCTAssertEqual([correctLayer1 count], [testedLayer1 count],
                   @"The single-neuron layer did not initialize with the correct number of elements.");
    XCTAssertEqual([correctLayer2 count], [testedLayer2 count],
                   @"The multi-neuron layer did not initialize with the correct number of elements.");
    
    BOOL correctLayer1Type = [[correctLayer1 objectAtIndex:0] class] == [[testedLayer1 objectAtIndex:0] class];
    BOOL correctLayer2Type_1 = [[correctLayer2 objectAtIndex:0] class] == [[testedLayer2 objectAtIndex:0] class];
    BOOL correctLayer2Type_2 = [[[correctLayer2 objectAtIndex:0] objectAtIndex:0] class] == [[[testedLayer2 objectAtIndex:0] objectAtIndex:0] class];
    
    //element type tests
    XCTAssertTrue(correctLayer1Type, @"The single-neuron layer elements are not the correct type.");
    XCTAssertTrue(correctLayer2Type_1, @"The multi-neuron layer elements are not the correct type.");
    XCTAssertTrue(correctLayer2Type_2, @"The multi-neuron slice elements are not the correct type.");
}

-(void)testTranspose {
    
    NSArray *before = @[@[@(1), @(2)], @[@(3), @(4)]];
    NSMutableArray *correctTransposed = [[NSMutableArray alloc] initWithArray:@[@[@(1), @(3)], @[@(2), @(4)]]];
    NSMutableArray *testedTransposed = [self.neuralNet transpose:before];
    
    XCTAssertEqualObjects(correctTransposed, testedTransposed, @"The transposed matrices do not match.");
    
}

-(void)testRandomWeight {
    
    double testedWeight = [self.neuralNet randomWeight];
    XCTAssertLessThan(-1.0, testedWeight, @"The random weight value is too low");
    XCTAssertGreaterThan(1.0, testedWeight, @"The random weight value is too high");
    
}

-(void)testSigmoid {
    double valueForSigmoid = 3.00;
    NSNumber *numberValue = @(valueForSigmoid);
    NSNumber *correctvalue = @(1.00 / (1.00 + exp((-numberValue.doubleValue))));
    NSNumber *testedValue = [self.neuralNet sigmoid:numberValue];
    
    XCTAssertEqualObjects(correctvalue, testedValue, @"The sigmoid values do not match.");
}

-(void)testVectorSigmoidConversion {
    
    NSMutableArray *before = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0), @(3.0)]];
    NSMutableArray *correctVector = [[NSMutableArray alloc] initWithArray:@[@(1.00 / (1.00 + exp((-1.0)))), @(1.00 / (1.00 + exp((-2.0)))), @(1.00 / (1.00 + exp((-3.0))))]];
    NSMutableArray *testVector = [self.neuralNet sigmoidForVector:before];
    
    XCTAssertEqualObjects(correctVector, testVector, @"The vector did not convert to sigmoid correctly.");
}

-(void)testMatrixSigmoidConversion {
    
    NSMutableArray *beforeSlice1 = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0), @(3.0)]];
    NSMutableArray *beforeSlice2 = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0), @(3.0)]];
    NSMutableArray *beforeMatrix = [[NSMutableArray alloc] initWithArray:@[beforeSlice1, beforeSlice2]];
    NSMutableArray *slice1 = [[NSMutableArray alloc] initWithArray:@[@(1.00 / (1.00 + exp((-1.0)))), @(1.00 / (1.00 + exp((-2.0)))), @(1.00 / (1.00 + exp((-3.0))))]];
    NSMutableArray *slice2 = [[NSMutableArray alloc] initWithArray:@[@(1.00 / (1.00 + exp((-1.0)))), @(1.00 / (1.00 + exp((-2.0)))), @(1.00 / (1.00 + exp((-3.0))))]];
    NSMutableArray *correctMatrix = [[NSMutableArray alloc] initWithArray:@[slice1, slice2]];
    NSMutableArray *testMatrix = [self.neuralNet sigmoidForMatrix:beforeMatrix];
    
    XCTAssertEqualObjects(correctMatrix, testMatrix, @"The matrix did not convert to sigmoid correctly.");
    
}

-(void)testDerivative {
    double valueForDerivative = 0.99;
    NSNumber *numberValue = @(valueForDerivative);
    NSNumber *correctValue = @(numberValue.doubleValue * (1.00 - numberValue.doubleValue));
    NSNumber *testedValue = [self.neuralNet derivative:numberValue];
    
    XCTAssertEqualObjects(correctValue, testedValue, @"The derivative values do not match.");
}

-(void)testVectorDerivativeConversion {
    
    NSMutableArray *before = [[NSMutableArray alloc] initWithArray:@[@(0.1), @(0.2), @(0.3)]];
    NSMutableArray *correctVector = [[NSMutableArray alloc] initWithArray:@[@(0.1 * (1.00 - 0.1)), @(0.2 * (1.00 - 0.2)), @(0.3 * (1.00 - 0.3))]];
    NSMutableArray *testVector = [self.neuralNet derivativeForVector:before];
    
    XCTAssertEqualObjects(correctVector, testVector, @"The vector did not convert to derivative correctly.");
}

-(void)testMatrixDerivativeConversion {
    
    NSMutableArray *beforeSlice1 = [[NSMutableArray alloc] initWithArray:@[@(0.1), @(0.2), @(0.3)]];
    NSMutableArray *beforeSlice2 = [[NSMutableArray alloc] initWithArray:@[@(0.4), @(0.5), @(0.6)]];
    NSMutableArray *beforeMatrix = [[NSMutableArray alloc] initWithArray:@[beforeSlice1, beforeSlice2]];
    NSMutableArray *slice1 = [[NSMutableArray alloc] initWithArray:@[@(0.1 * (1.00 - 0.1)), @(0.2 * (1.00 - 0.2)), @(0.3 * (1.00 - 0.3))]];
    NSMutableArray *slice2 = [[NSMutableArray alloc] initWithArray:@[@(0.4 * (1.00 - 0.4)), @(0.5 * (1.00 - 0.5)), @(0.6 * (1.00 - 0.6))]];
    NSMutableArray *correctMatrix = [[NSMutableArray alloc] initWithArray:@[slice1, slice2]];
    NSMutableArray *testMatrix = [self.neuralNet derivativeForMatrix:beforeMatrix];
    
    XCTAssertEqualObjects(correctMatrix, testMatrix, @"The matrix did not convert to derivative correctly.");
    
}

-(void)testAddVectorElements {
    
    NSArray *vector1 = @[@(1.0), @(2.0), @(3.0), @(4.0)];
    NSArray *vector2 = @[@(5.2), @(6.2), @(7.2), @(8.2)];
    NSMutableArray *correctVector = [[NSMutableArray alloc] initWithArray:@[@(6.2), @(8.2), @(10.2), @(12.2)]];
    NSMutableArray *testVector = [self.neuralNet addVector:vector1 toVector:vector2];
    
    XCTAssertEqualObjects(correctVector, testVector, @"Vector element-wise addition failed.");
}

-(void)testAddMatrixElements {
    
    NSArray *matrix1 = @[@[@(1.0), @(2.0)], @[@(3.0), @(4.0)]];
    NSArray *matrix2 = @[@[@(5.2), @(6.2)], @[@(7.2), @(8.2)]];
    NSMutableArray *correctMatrix = [[NSMutableArray alloc] initWithArray:@[@[@(6.2), @(8.2)], @[@(10.2), @(12.2)]]];
    NSMutableArray *testMatrix = [self.neuralNet addMatrix:matrix1 toMatrix:matrix2];
    
    XCTAssertEqualObjects(correctMatrix, testMatrix, @"Matrix element-wise addition failed.");
    
}

-(void)testMultiplyVectorElements {
    
    NSArray *vector1 = @[@(1.0), @(2.0), @(3.0), @(4.0)];
    NSArray *vector2 = @[@(5.2), @(6.2), @(7.2), @(8.2)];
    NSMutableArray *correctVector = [[NSMutableArray alloc] initWithArray:@[@(5.2), @(12.4), @(21.6), @(32.8)]];
    NSMutableArray *testVector = [self.neuralNet multiplyVectorElements:vector1 by:vector2];
    
    XCTAssertEqualObjects(correctVector, testVector, @"Vector element-wise multiplication failed.");
}

-(void)testOuterProduct {
    
    NSMutableArray *matrix1 = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0), @(3.0)]];
    NSMutableArray *matrix2 = [[NSMutableArray alloc] initWithArray:@[@(4.2), @(5.2), @(6.2)]];
    
    NSMutableArray *layer1 = [[NSMutableArray alloc] initWithArray:@[@(4.2), @(5.2), @(6.2)]];
    NSMutableArray *layer2 = [[NSMutableArray alloc] initWithArray:@[@(8.4), @(10.4), @(12.4)]];
    NSMutableArray *layer3 = [[NSMutableArray alloc] initWithArray:@[@(12.6), @(15.6), @(18.6)]];
    NSMutableArray *correctMatrix = [[NSMutableArray alloc]
                                     initWithArray:@[layer1, layer2, layer3]];
    
    NSMutableArray *testMatrix = [self.neuralNet outerProduct:matrix1 by:matrix2];
    
    for (int i = 0; i < [correctMatrix count]; i++) {
        NSArray *slice1 = [correctMatrix objectAtIndex:i];
        NSArray *slice2 = [testMatrix objectAtIndex:i];
        for (int j = 0; j < [slice1 count]; j++) {
            [self compareNumberWithPrecision:[slice1 objectAtIndex:j] to:[slice2 objectAtIndex:j]];
            XCTAssertEqualWithAccuracy([[slice1 objectAtIndex:j] doubleValue],
                                       [[slice1 objectAtIndex:j] doubleValue],
                                       0.000000000000001, @"Matrix outer product failed");
        }
    }
}

-(void)testMatrixVectorProduct {
    
    NSMutableArray *vector = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0)]];
    NSMutableArray *matrix = [[NSMutableArray alloc] initWithArray:@[@[@(4.2), @(5.2)],
                                                                     @[@(6.2), @(7.2)]]];
    
    NSMutableArray *correctMatrix = [[NSMutableArray alloc] initWithArray:@[@(14.6), @(20.6)]];
    NSMutableArray *testMatrix = [self.neuralNet dotProductMatrix:matrix byVector:vector];
    
    for (int i = 0; i < [correctMatrix count]; i++) {
        [self compareNumberWithPrecision:[correctMatrix objectAtIndex:i] to:[testMatrix objectAtIndex:i]];
        XCTAssertEqualWithAccuracy([[correctMatrix objectAtIndex:i] doubleValue],
                                       [[testMatrix objectAtIndex:i] doubleValue],
                                       0.00000000000001, @"Matrix-vector product failed.");
    }
    
    //XCTAssertEqualObjects(correctMatrix, testMatrix, @"The matrix-vector product calculation failed.");
    
}

-(void)testVectorDotProduct {
    
    NSArray *vector1 = [[NSMutableArray alloc] initWithArray:@[@(1.0), @(2.0), @(3.0)]];
    NSArray *vector2 = [[NSMutableArray alloc] initWithArray:@[@(4.0), @(5.0), @(6.0)]];
    NSMutableArray *throwTest = [[NSMutableArray alloc] initWithArray:@[@(7.0), @(8.0)]];
    double correctDotProduct = 32.0;
    double testDotProduct = [self.neuralNet vectorDotProduct:vector1 by:vector2];
    
    XCTAssertEqual(correctDotProduct, testDotProduct, @"The vector dot product did not calculate correctly.");
    
    XCTAssertThrows([self.neuralNet vectorDotProduct:vector1 by:throwTest]);
}

-(void)testMatrixDotProduct {
    
    NSArray *matrix1 = @[@[@(1.0), @(2.0)],
                        @[@(3.1), @(4.2)]];
    NSArray *matrix2 = @[@[@(3.0), @(4.0)],
                         @[@(5.1), @(6.2)]];
    
    NSArray *correctMatrix = @[@[@(13.2), @(16.4)],
                                @[@(30.72), @(38.44)]
                                ];
    NSMutableArray *testMatrix = [self.neuralNet dotProduct:matrix1 by:matrix2];
    
    for (int i = 0; i < [correctMatrix count]; i++) {
        NSArray *slice1 = [correctMatrix objectAtIndex:i];
        NSArray *slice2 = [testMatrix objectAtIndex:i];
        for (int j = 0; j < [slice1 count]; j++) {
            [self compareNumberWithPrecision:[slice1 objectAtIndex:j] to:[slice2 objectAtIndex:j]];
            XCTAssertEqualWithAccuracy([[slice1 objectAtIndex:j] doubleValue],
                                       [[slice1 objectAtIndex:j] doubleValue],
                                       0.000000000000001, @"Matrix dot product failed.");
        }
    }
}

-(void)testMultiplyMatrixElements {
    
    NSArray *matrix1 = @[@[@(1.0), @(2.0)],
                         @[@(3.1), @(4.2)]];
    NSArray *matrix2 = @[@[@(3.0), @(4.0)],
                         @[@(5.1), @(6.2)]];
    
    NSArray *correctMatrix = @[@[@(3.0), @(8.0)],
                               @[@(15.81), @(26.04)]
                               ];
    
    NSArray *testMatrix = [self.neuralNet multiplyMatrixElements:matrix1 by:matrix2];
    
    for (int i = 0; i < [correctMatrix count]; i++) {
        NSArray *slice1 = [correctMatrix objectAtIndex:i];
        NSArray *slice2 = [testMatrix objectAtIndex:i];
        for (int j = 0; j < [slice1 count]; j++) {
            [self compareNumberWithPrecision:[slice1 objectAtIndex:j] to:[slice2 objectAtIndex:j]];
            XCTAssertEqualWithAccuracy([[slice1 objectAtIndex:j] doubleValue],
                                       [[slice1 objectAtIndex:j] doubleValue],
                                       0.000000000000001, @"Matrix dot product failed.");
        }
    }
}

#pragma mark - NSNumber Floating Point Inspection Helper

-(void)compareNumberWithPrecision:(NSNumber *)number1 to:(NSNumber *)number2 {
    NSLog(@"First Value: %.16lf Second Value:%.16lf", [number1 doubleValue], [number2 doubleValue]);
}

@end
