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
    
    NSArray *training = @[@[@(0), @(0), @(1)],
                          @[@(0), @(1), @(1)],
                          @[@(1), @(0), @(1)],
                          @[@(0), @(1), @(0)],
                          @[@(1), @(0), @(0)],
                          @[@(1), @(1), @(1)],
                          @[@(0), @(0), @(0)]
                          ];
    
    NSArray *trainingOutput = @[@(0), @(1), @(1), @(1), @(1), @(0), @(0)];
    
    [self.neuralNet train:training trainingOutput:trainingOutput iterations:60000];
    [self.neuralNet predict:@[@(0), @(1), @(0)]];
}

@end
