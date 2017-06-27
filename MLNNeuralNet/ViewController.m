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

@property (strong, nonatomic) UILabel *trainButton;
@property BOOL isTraining;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    //Initialize a neural net
    _neuralNet = [[MLNNeuralNet alloc] initWithInputs:3 hiddenSize:4];
    
    
    //Description label
    UILabel *description =[[UILabel alloc] init];
    description.frame = CGRectMake(self.view.frame.size.width * 0.1, self.view.frame.size.height * 0.1, self.view.frame.size.width * 0.8, self.view.frame.size.height * 0.3);
    description.numberOfLines = 0;
    description.text = @"This is a simple example that trains a two-layer neural net to learn a XOR gate at the first two inputs. Press train to start and view results in the console.";
    [self.view addSubview:description];
    
    //Create a button to start training
    _trainButton = [[UILabel alloc] initWithFrame:CGRectMake(self.view.frame.size.width * 0.1, self.view.frame.size.height * 0.4, self.view.frame.size.width * 0.8, self.view.frame.size.height * 0.05)];
    self.trainButton.backgroundColor = [UIColor blueColor];
    self.trainButton.textColor = [UIColor whiteColor];
    self.trainButton.font = [UIFont boldSystemFontOfSize:22];
    self.trainButton.text = @"Train";
    self.trainButton.textAlignment = NSTextAlignmentCenter;
    self.trainButton.userInteractionEnabled = YES;
    self.trainButton.layer.cornerRadius = 4;
    self.trainButton.layer.masksToBounds = YES;
    self.trainButton.layer.borderColor = [UIColor blackColor].CGColor;
    [self.view addSubview:self.trainButton];
    
    UITapGestureRecognizer *train = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(start)];
    [self.trainButton addGestureRecognizer:train];
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(void)start {
    
    if (self.isTraining) {
        return;
    }
    else {
        self.isTraining = YES;
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
        
        
        //Train the network asynchronously
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            NSLog(@"Training neural net...");
            [self.neuralNet train:training trainingOutput:trainingOutput iterations:10000];
            self.isTraining = NO;
            
            dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
                NSLog(@"Predicting from input...");
                [self.neuralNet predict:@[@(0), @(1), @(0)]];
            });
        });
    }

}

@end
