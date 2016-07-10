//
//  Poet.swift
//
//  Created by Mcmahon, Craig on 7/1/16.
//  Copyright © 2016 handform. All rights reserved.
//

import BrainCore
import Upsurge
import HDF5Kit
import Metal

func sample(output: ValueArray<Float>, temperature: Float) -> Int {
    var a = log(output) / temperature
    a = exp(a) / sum(exp(a))
    
    while true {
        for (index, prob) in a.enumerate() {
            let random = Float(arc4random()) / 0xFFFFFFFF
            if random < prob {
                return index
            }
        }
    }
}


/// This class generates characters from weights trained by the Keras example `lstm_text_generation.py`.
class Poet {

    // Obtain this array by adding the line `print('chars: ', chars)` to `lstm_text_generation.py`.
    let chars = ["\n", " ", "!", "\"", "\"", "(", ")", "*", ",", "-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "?", "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "?", "?", "?", "?", "?"]
    let unitCount = 512
    let batchSize = 1

    lazy var inputSize: Int = { self.chars.count }()
    
    var pathToTrainedWeights: String
    var dataLayer: Source?
    var sinkLayer: Sink?
    var net: Net?
    var evaluator: Evaluator?
    var temperature: Float = 0.5
    let semaphore = dispatch_semaphore_create(0)

    var device: MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            fatalError("Failed to create a Metal device")
        }
        return d
    }
    
    init(pathToTrainedWeights: String) {
        self.pathToTrainedWeights = pathToTrainedWeights
    }

    private func inputFromChar(char: String) -> Matrix<Float> {
        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            // "One hot" input with '1' at the index of the character we will pass in
            input[0, i] = (chars[i] == char) ? 1 : 0
        }
        return input
    }

    func prepareToEvaluate(completion: (prepared: Bool) -> ()) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0)) {
            let builder = NetworkBuilder()
            self.net = builder.loadNetFromFile(self.pathToTrainedWeights)
            self.dataLayer = builder.dataLayer     // input
            self.sinkLayer = builder.sinkLayer    // output
            completion(prepared: true)
        }
    }
    
    func startEvaluating(seed seed: String, callback: (string: String) -> ()) -> Bool {
        guard let net = self.net else {
            return false
        }
        
        do {
            evaluator = try Evaluator(net: net, device: self.device)
        } catch {
            return false
        }
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0)) {
            self.evaluate(seed: seed, callback: callback)
        }

        return true
    }

    func evaluate(seed seed: String, callback: (string: String) -> ()) {
        guard let dataLayer = self.dataLayer, sinkLayer = self.sinkLayer, semaphore = self.semaphore else {
            fatalError("Not initialized")
        }
        guard let evaluator = self.evaluator else {
            fatalError("Not initialized")
        }

        // Seed
        let input = ValueArray<Float>(count: NetworkBuilder.inputSize, repeatedValue: 0)
        for c in seed.characters {
            for i in 0..<NetworkBuilder.inputSize {
                input[i] = 0
            }
            if let index = chars.indexOf(String(c)) {
                input[index] = 1
            }
            dataLayer.data = input
            evaluator.evaluate { _ in
                // Ignore output
                dispatch_semaphore_signal(semaphore)
            }
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)
        }

        // Run
        while true {

            evaluator.evaluate { (snapshot) in
                let output = sinkLayer.data

                let exps = output.map(expf)
                let sum  = exps.reduce(0, combine: +)
                let softmax = exps / sum

                let index = sample(softmax, temperature: self.temperature)

                let char = self.chars[index]
                // print(self.chars[maxIndex], terminator: "")

                callback(string: char)

                dataLayer.data = self.inputFromChar(char).elements

                dispatch_semaphore_signal(semaphore)
            }
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)
        }
    }

    func stopEvaluating() {
        self.evaluator = nil
    }
    
}
