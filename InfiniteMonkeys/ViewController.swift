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
    @IBOutlet var startStopButton: NSButton!
    @IBOutlet var temperatureSlider: NSSlider!
    
    var poet: Poet?

    @IBAction func clickedStartStopButton(sender: AnyObject?) {
        if let poet = self.poet {
            poet.stopEvaluating()
            startStopButton.title = "Start"
            self.poet = nil
            return
        }

        startStopButton.enabled = false
        startStopButton.title = "Stop"
        
        self.textView.textStorage?.setAttributedString(NSAttributedString(string: "...", attributes: [NSFontAttributeName: self.textView.font!]))

        if let
            path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5"),
            poet = Poet(pathToTrainedWeights: path)
        {
            poet.temperature = temperatureSlider.floatValue

            poet.prepareToEvaluate(
                seed: "e",
                completion: { (prepared) in
                    var count = 0
                    
                    poet.startEvaluating({ (string) in
                        dispatch_async(dispatch_get_main_queue()) {
                            self.startStopButton.enabled = true
                            
                            count += 1
                            let scroll = abs(NSMaxY(self.textView.visibleRect) - NSMaxY(self.textView.bounds)) < 50
                            self.textView.textStorage?.mutableString.appendString(string)
                            if scroll {
                                self.textView.scrollRangeToVisible(NSMakeRange(count, 0))
                            }
                        }
                    })
            })

            self.poet = poet
        }
    }
    
    @IBAction func adjustedTemperatureSlider(sender: AnyObject?) {
        poet?.temperature = temperatureSlider.floatValue
    }

}

