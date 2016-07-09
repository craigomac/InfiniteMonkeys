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
    
    var weightsFile: File?
    var dataLayer: Source?
    var denseLayer: InnerProductLayer?
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
    
    class Source: DataLayer {
        let name: String?
        let id = NSUUID()
        var data: Blob
        var batchSize: Int
        
        var outputSize: Int {
            return data.count / batchSize
        }
        
        init(name: String, data: Blob, batchSize: Int) {
            self.name = name
            self.data = data
            self.batchSize = batchSize
        }
        
        func nextBatch(batchSize: Int) -> Blob {
            return data
        }
    }
    
    class Sink: SinkLayer {
        let name: String?
        let id = NSUUID()
        var inputSize: Int
        var batchSize: Int
        
        var data: Blob = []
        
        init(name: String, inputSize: Int, batchSize: Int) {
            self.name = name
            self.inputSize = inputSize
            self.batchSize = batchSize
        }
        
        func consume(input: Blob) {
            self.data = input
        }
    }
    
    init?(pathToTrainedWeights: String) {
        weightsFile = File.open(pathToTrainedWeights, mode: .ReadOnly)
    }

    private func inputFromChar(char: String) -> Matrix<Float> {
        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            // "One hot" input with '1' at the index of the character we will pass in
            input[0, i] = (chars[i] == char) ? 1 : 0
        }
        return input
    }

    func prepareToEvaluate(seed seed: String, completion: (prepared: Bool) -> ()) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0)) {
            guard let file = self.weightsFile else {
                completion(prepared: false)
                return
            }
            
            let unitCount = self.unitCount, inputSize = self.inputSize, batchSize = self.batchSize
            
            do {
                guard let
                    lstm_1 = file.openGroup("lstm_1"),
                    lstm_1_U_c = try lstm_1.openFloatDataset("lstm_1_U_c")?.read(),
                    lstm_1_U_f = try lstm_1.openFloatDataset("lstm_1_U_f")?.read(),
                    lstm_1_U_i = try lstm_1.openFloatDataset("lstm_1_U_i")?.read(),
                    lstm_1_U_o = try lstm_1.openFloatDataset("lstm_1_U_o")?.read(),
                    lstm_1_W_c = try lstm_1.openFloatDataset("lstm_1_W_c")?.read(),
                    lstm_1_W_f = try lstm_1.openFloatDataset("lstm_1_W_f")?.read(),
                    lstm_1_W_i = try lstm_1.openFloatDataset("lstm_1_W_i")?.read(),
                    lstm_1_W_o = try lstm_1.openFloatDataset("lstm_1_W_o")?.read(),
                    lstm_1_b_c = try lstm_1.openFloatDataset("lstm_1_b_c")?.read(),
                    lstm_1_b_f = try lstm_1.openFloatDataset("lstm_1_b_f")?.read(),
                    lstm_1_b_i = try lstm_1.openFloatDataset("lstm_1_b_i")?.read(),
                    lstm_1_b_o = try lstm_1.openFloatDataset("lstm_1_b_o")?.read()
                else {
                    completion(prepared: false)
                    return
                }
                
                guard let
                    lstm_2 = file.openGroup("lstm_2"),
                    lstm_2_U_c = try lstm_2.openFloatDataset("lstm_2_U_c")?.read(),
                    lstm_2_U_f = try lstm_2.openFloatDataset("lstm_2_U_f")?.read(),
                    lstm_2_U_i = try lstm_2.openFloatDataset("lstm_2_U_i")?.read(),
                    lstm_2_U_o = try lstm_2.openFloatDataset("lstm_2_U_o")?.read(),
                    lstm_2_W_c = try lstm_2.openFloatDataset("lstm_2_W_c")?.read(),
                    lstm_2_W_f = try lstm_2.openFloatDataset("lstm_2_W_f")?.read(),
                    lstm_2_W_i = try lstm_2.openFloatDataset("lstm_2_W_i")?.read(),
                    lstm_2_W_o = try lstm_2.openFloatDataset("lstm_2_W_o")?.read(),
                    lstm_2_b_c = try lstm_2.openFloatDataset("lstm_2_b_c")?.read(),
                    lstm_2_b_f = try lstm_2.openFloatDataset("lstm_2_b_f")?.read(),
                    lstm_2_b_i = try lstm_2.openFloatDataset("lstm_2_b_i")?.read(),
                    lstm_2_b_o = try lstm_2.openFloatDataset("lstm_2_b_o")?.read()
                else {
                    completion(prepared: false)
                    return
                }
                
                let layer1_weights = LSTMLayer.makeWeightsFromComponents(
                    Wc: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_c),
                    Wf: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_f),
                    Wi: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_i),
                    Wo: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_o),
                    Uc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_c),
                    Uf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_f),
                    Ui: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_i),
                    Uo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_o))
                
                guard let
                    dense_1 = file.openGroup("dense_1"),
                    dense_1_W = try dense_1.openFloatDataset("dense_1_W")?.read(),
                    dense_1_b = try dense_1.openFloatDataset("dense_1_b")?.read()
                else {
                    completion(prepared: false)
                    return
                }
                
                let layer2_weights = LSTMLayer.makeWeightsFromComponents(
                    Wc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_c),
                    Wf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_f),
                    Wi: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_i),
                    Wo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_o),
                    Uc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_c),
                    Uf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_f),
                    Ui: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_i),
                    Uo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_o))
                
                let layer1 = LSTMLayer(
                    weights: layer1_weights,
                    biases: ValueArray([ lstm_1_b_i, lstm_1_b_c, lstm_1_b_f, lstm_1_b_o ].flatMap({ $0 })),
                    batchSize: batchSize,
                    name: "lstm_1")
                
                
                
                let layer2 = LSTMLayer(
                    weights: layer2_weights,
                    biases: ValueArray([ lstm_2_b_i, lstm_2_b_c, lstm_2_b_f, lstm_2_b_o ].flatMap({ $0 })),
                    batchSize: batchSize,
                    name: "lstm_2")
                
                let denseLayer = InnerProductLayer(weights: Matrix(rows: unitCount, columns: inputSize, elements: dense_1_W),
                                                   biases: ValueArray(dense_1_b))

                let input = self.inputFromChar(seed)

                let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
                let sink = Sink(name: "sink", inputSize: inputSize, batchSize: 1)

                self.dataLayer = dataLayer      // input
                self.denseLayer = denseLayer    // output

                self.net = Net.build {
                    dataLayer => layer1 => layer2 => denseLayer => sink
                }
            } catch {
                completion(prepared: false)
                return
            }
            
            completion(prepared: true)
        }
    }
    
    func startEvaluating(callback: (string: String) -> ()) -> Bool {
        guard let net = self.net, dataLayer = self.dataLayer, denseLayer = self.denseLayer, semaphore = self.semaphore else {
            return false
        }
        
        do {
            self.evaluator = try Evaluator(net: net, device: self.device)
        } catch {
            return false
        }
        
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0)) {

            while true {
                guard let evaluator = self.evaluator else {
                    return
                }

                evaluator.evaluate { (snapshot) in
                    let output = [Float](snapshot.outputOfLayer(denseLayer)!)
                    
                    let exps = output.map(expf)
                    let sum  = exps.reduce(0, combine: +)
                    let softmax = exps / sum
                    
                    let index = sample(softmax, temperature: self.temperature)
                    
                    let char = self.chars[index]
                    // print(self.chars[maxIndex], terminator: "")
                    
                    callback(string: char)

                    dataLayer.data = self.inputFromChar(char).elements

                    dispatch_semaphore_signal(semaphore);
                }
                dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)
            }
        }

        return true
    }
    
    func stopEvaluating() {
        self.evaluator = nil
    }
    
}
