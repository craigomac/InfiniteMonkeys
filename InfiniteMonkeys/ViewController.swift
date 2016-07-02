//
//  ViewController.swift
//  InfiniteMonkeys
//
//  Created by Mcmahon, Craig on 7/1/16.
//  Copyright © 2016 handform. All rights reserved.
//

import Cocoa

class ViewController: NSViewController {

    override func viewDidLoad() {
        super.viewDidLoad()

        if let
            path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5"),
            poet = Poet(pathToTrainedWeights: path)
        {
            poet.prepareToEvaluate(
                seed: "e",
                completion: { (prepared) in
                    poet.startEvaluating()
            })
        }
    }

    override var representedObject: AnyObject? {
        didSet {
        // Update the view, if already loaded.
        }
    }
    
    

}

