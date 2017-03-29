//
//  ViewController.m
//  MLNNeuralNet
//
//  Created by Jason Dwyer on 6/7/16.
//  Copyright © 2016 Jason Dwyer. All rights reserved.
//

#import "ViewController.h"
#import "MLNNeuralNet.h"

@interface ViewController ()

@property (strong, nonatomic) MLNNeuralNet *neuralNet;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    _neuralNet = [[MLNNeuralNet alloc] initWithInputs:3 hiddenSize:4];
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    
    /*Here we override touchesBegan to initiate the neural net training and prediction*/
    
    /*Create a matrix of inputs for demo purpose
     This example tests if the neural net can learn the XOR gate in the first 2 inputs
     */
    NSArray *training = @[@[@(0), @(0), @(1)],
                          @[@(0), @(1), @(1)],
                          @[@(1), @(0), @(1)],
                          @[@(0), @(1), @(0)],
                          @[@(1), @(0), @(0)],
                          @[@(1), @(1), @(1)],
                          @[@(0), @(0), @(0)]
                          ];
    
    //Create output for demo purposes
    NSArray *trainingOutput = @[@(0), @(1), @(1), @(1), @(1), @(0), @(0)];
    
    
    MLNNeuralNet *newNet = [MLNNeuralNet neuralNetWithInputs:4];
    NSLog(@"Inputs: %lu", (unsigned long)[newNet.wxh count]);
    NSLog(@"Hidden: %lu", (unsigned long)[newNet.why count]);
    
    [self.neuralNet train:training trainingOutput:trainingOutput iterations:60000];
    [self.neuralNet predict:@[@(0), @(1), @(0)]];
}

@end
