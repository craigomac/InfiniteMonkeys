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

        guard let path = Bundle.main.path(forResource: "lstm_text_generation_weights", ofType: "h5") else {
            fatalError("Weigths file not found")
        }
        poet = Poet(pathToTrainedWeights: path)

        #if arch(i386) || arch(x86_64)
            preconditionFailure("This app will not function on the iOS Simulator because of Metal-dependent functionality.")
        #endif
    }

    @IBAction func tappedStartStopButton(_ sender: AnyObject?) {
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
        startStopButton.setTitle("Stop", for: UIControlState())

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
        poet.startEvaluating(seedTextField.text!) { string in
            buffer = buffer + string
            count += 1

            if count > 5 {
                let bufferCopy = buffer
                DispatchQueue.main.async {
                    self.textView.text = self.textView.text + bufferCopy
                }

                count = 0
                buffer = ""
            }
        }
    }

    func prepare(_ completion: @escaping () -> Void) {
        textView.text = "Loading..."
        startStopButton.isEnabled = false
        poet?.prepareToEvaluate { prepared in
            DispatchQueue.main.async {
                self.startStopButton.isEnabled = true
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
        startStopButton.setTitle("Start", for: UIControlState())
    }

    func disableControls() {
        seedTextField.resignFirstResponder()
        seedTextField.isEnabled = false
        diversitySlider.isEnabled = false
    }

    func enableControls() {
        seedTextField.isEnabled = true
        diversitySlider.isEnabled = true
        startStopButton.isEnabled = true
    }

}

