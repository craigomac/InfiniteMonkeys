//
//  ViewController.swift
//  InfiniteMonkeys
//
//  Created by Mcmahon, Craig on 7/1/16.
//  Copyright © 2016 handform. All rights reserved.
//

import Cocoa

class ViewController: NSViewController {

    @IBOutlet var textView: NSTextView!
    
    override func viewDidLoad() {
        super.viewDidLoad()

        if let
            path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5"),
            poet = Poet(pathToTrainedWeights: path)
        {
            poet.prepareToEvaluate(
                seed: "e",
                completion: { (prepared) in
                    
                    self.textView.textStorage?.setAttributedString(NSAttributedString(string: "...", attributes: [NSFontAttributeName: self.textView.font!]))
                    var count = 0
                    
                    poet.startEvaluating({ (string) in
                        dispatch_async(dispatch_get_main_queue()) {
                            count += 1
                            let scroll = abs(NSMaxY(self.textView.visibleRect) - NSMaxY(self.textView.bounds)) < 50
                            self.textView.textStorage?.mutableString.appendString(string)
                            if scroll {
                                self.textView.scrollRangeToVisible(NSMakeRange(count, 0))
                            }
                        }
                    })
            })
        }
    }

    override var representedObject: AnyObject? {
        didSet {
        // Update the view, if already loaded.
        }
    }
    
    

}

