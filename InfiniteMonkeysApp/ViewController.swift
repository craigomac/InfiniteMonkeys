//
//  ViewController.swift
//  InfiniteMonkeysApp
//
//  Created by Mcmahon, Craig on 7/9/16.
//  Copyright © 2016 handform. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var seedTextField: UITextField!
    @IBOutlet weak var diversitySlider: UISlider!
    @IBOutlet weak var startStopButton: UIButton!
    @IBOutlet weak var textView: UITextView!
    
    var poet: Poet?
    
    override func viewDidLoad() {
        super.viewDidLoad()

        guard let path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5") else {
            fatalError("Weigths file not found")
        }
        poet = Poet(pathToTrainedWeights: path)

        #if arch(i386) || arch(x86_64)
            preconditionFailure("This app will not function on the iOS Simulator because of Metal-dependent functionality.")
        #endif
    }

    @IBAction func tappedStartStopButton(sender: AnyObject?) {
        if poet?.isEvaluating ?? false {
            stop()
        } else {
            start()
        }
    }

    func start() {
        guard let poet = poet else {
            return
        }

        disableControls()
        startStopButton.setTitle("Stop", forState: .Normal)

        if !poet.isPrepared {
            prepare {
                self.start()
            }
            return
        }

        var buffer: String = ""
        var count = 0

        textView.text = seedTextField.text
        poet.temperature = diversitySlider.value
        poet.startEvaluating(seed: seedTextField.text!) { string in
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
        }
    }

    func prepare(completion: () -> Void) {
        textView.text = "Loading..."
        startStopButton.enabled = false
        poet?.prepareToEvaluate { prepared in
            dispatch_async(dispatch_get_main_queue()) {
                self.startStopButton.enabled = true
                if prepared {
                    completion()
                } else {
                    self.textView.text = nil
                }
            }
        }
    }

    func stop() {
        poet?.stopEvaluating()
        enableControls()
        startStopButton.setTitle("Start", forState: .Normal)
    }

    func disableControls() {
        seedTextField.resignFirstResponder()
        seedTextField.enabled = false
        diversitySlider.enabled = false
    }

    func enableControls() {
        seedTextField.enabled = true
        diversitySlider.enabled = true
        startStopButton.enabled = true
    }

}

