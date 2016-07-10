//
//  ViewController.swift
//  InfiniteMonkeysApp
//
//  Created by Mcmahon, Craig on 7/9/16.
//  Copyright © 2016 handform. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var startStopButton: UIButton!
    @IBOutlet weak var textView: UITextView!
    
    var poet: Poet?
    
    override func viewDidLoad() {
        #if arch(i386) || arch(x86_64)
            assert(false, "This app will not function on the iOS Simulator because of Metal-dependent functionality.")
        #endif
    }

    @IBAction func tappedStartStopButton(sender: AnyObject?) {
        if let poet = self.poet {
            poet.stopEvaluating()
            startStopButton.setTitle("Start", forState: .Normal)
            self.poet = nil
            return
        }
        
        startStopButton.enabled = false
        startStopButton.setTitle("Stop", forState: .Normal)
        
        self.textView.text = "..."
        
        var buffer: String = ""
        var count = 0
        
        if let path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5") {
            poet = Poet(pathToTrainedWeights: path)
            poet!.prepareToEvaluate(
                seed: "e",
                completion: { (prepared) in
                    dispatch_async(dispatch_get_main_queue()) {
                        self.startStopButton.enabled = true
                    }

                    self.poet!.startEvaluating({ (string) in
                        buffer = buffer + string
                        count += 1

                        if count > 5 {
                            let bufferCopy = buffer
                            dispatch_async(dispatch_get_main_queue()) {
                                self.textView.text = self.textView.text + bufferCopy
                            }

                            count = 0
                            buffer = ""
                        }
                    })
            })

        }
    }


}

