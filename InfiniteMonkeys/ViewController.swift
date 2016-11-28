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
    @IBOutlet var seedField: NSTextField!
    @IBOutlet var startStopButton: NSButton!
    @IBOutlet var temperatureSlider: NSSlider!
    
    var poet: Poet?

    @IBAction func clickedStartStopButton(_ sender: AnyObject?) {
        if let poet = self.poet {
            poet.stopEvaluating()
            startStopButton.title = "Start"
            self.poet = nil
            return
        }

        startStopButton.isEnabled = false
        startStopButton.title = "Stop"
        
        self.textView.textStorage?.setAttributedString(NSAttributedString(string: "...", attributes: [NSFontAttributeName: self.textView.font!]))

        if let path = Bundle.main.path(forResource: "lstm_text_generation_weights", ofType: "h5") {
            poet = Poet(pathToTrainedWeights: path)
            
            guard let poet = poet else {
                // Error alert
                return
            }

            poet.temperature = temperatureSlider.floatValue

            poet.prepareToEvaluate { (prepared) in
                self.startStopButton.isEnabled = true

                let seed = self.seedField.stringValue
                self.textView.textStorage?.setAttributedString(NSAttributedString(string: seed, attributes: [NSFontAttributeName: self.textView.font!]))

                var count = 0
                var buffer = ""

                _ = self.poet?.startEvaluating(seed) { (string) in
                    buffer += string
                    count += 1

                    if string == " " {
                        let bufferCopy = buffer
                        let countCopy = count

                        DispatchQueue.main.async {
                            let scroll = abs(NSMaxY(self.textView.visibleRect) - NSMaxY(self.textView.bounds)) < 50
                            self.textView.textStorage?.mutableString.append(bufferCopy)
                            if scroll {
                                self.textView.scrollRangeToVisible(NSMakeRange(countCopy, 0))
                            }
                        }

                        buffer = ""
                    }
                }
            }
        }
    }
    
    @IBAction func adjustedTemperatureSlider(_ sender: AnyObject?) {
        temperatureSlider.toolTip = "\(temperatureSlider.floatValue)"
        poet?.temperature = temperatureSlider.floatValue
    }

}

