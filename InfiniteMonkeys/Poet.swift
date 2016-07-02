//
//  Poet.swift
//
//  Created by Mcmahon, Craig on 7/1/16.
//  Copyright © 2016 handform. All rights reserved.
//

import BrainCore
import Upsurge
import HDF5Kit

extension ValueArraySlice {
    
    mutating func reset() -> ValueArraySlice {
        endIndex = endIndex - startIndex
        startIndex = 0
        return self
    }
    
}

class Poet {

    func test() throws {
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
        
        guard let
            path = NSBundle.mainBundle().pathForResource("lstm_text_generation_weights", ofType: "h5"),
            file = File.open(path, mode: .ReadOnly)
            else {
                return
        }
        
        do {
            for layerName in ["lstm_1", "lstm_2", "dense_1", "dropout_1", "activation_1"] {
                if let layer = file.openGroup(layerName) {
                    for objectName in layer.objectNames() {
                        if let
                            floats = try layer.openFloatDataset(objectName)?.read()
                        {
                            print("\(objectName): \(floats.count)\((((floats.count != 512) && (floats.count%512==0)) ? (" (\(floats.count/512) * 512)") : "")) values")
                        }
                    }
                }
            }
        } catch {
            
        }
        
        let inputSize = 57
        let unitCount = 512
        let batchSize = 1
        
        let chars = ["\n", " ", "!", "\"", "\"", "(", ")", "*", ",", "-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "?", "[", "]", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "?", "?", "?", "?", "?"]
        
        assert(chars.count == inputSize)
        
        guard let
            layer1 = file.openGroup("lstm_1"),
            lstm_1_U_c = try layer1.openFloatDataset("lstm_1_U_c")?.read(),
            lstm_1_U_f = try layer1.openFloatDataset("lstm_1_U_f")?.read(),
            lstm_1_U_i = try layer1.openFloatDataset("lstm_1_U_i")?.read(),
            lstm_1_U_o = try layer1.openFloatDataset("lstm_1_U_o")?.read(),
            lstm_1_W_c = try layer1.openFloatDataset("lstm_1_W_c")?.read(),
            lstm_1_W_f = try layer1.openFloatDataset("lstm_1_W_f")?.read(),
            lstm_1_W_i = try layer1.openFloatDataset("lstm_1_W_i")?.read(),
            lstm_1_W_o = try layer1.openFloatDataset("lstm_1_W_o")?.read(),
            lstm_1_b_c = try layer1.openFloatDataset("lstm_1_b_c")?.read(),
            lstm_1_b_f = try layer1.openFloatDataset("lstm_1_b_f")?.read(),
            lstm_1_b_i = try layer1.openFloatDataset("lstm_1_b_i")?.read(),
            lstm_1_b_o = try layer1.openFloatDataset("lstm_1_b_o")?.read()
            else {
                return
        }
        
        let layer1_weights = Poet.makeWeightsFromComponents(
            Wc: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_c),
            Wf: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_f),
            Wi: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_i),
            Wo: Matrix(rows: inputSize, columns: unitCount, elements: lstm_1_W_o),
            Uc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_c),
            Uf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_f),
            Ui: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_i),
            Uo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_1_U_o))
        
        let lstm_1 = LSTMLayer(
            weights: layer1_weights,
            biases: ValueArray([ lstm_1_b_i, lstm_1_b_c, lstm_1_b_f, lstm_1_b_o ].flatMap({ $0 })),
            batchSize: batchSize,
            name: "lstm_1")
        
        guard let
            layer2 = file.openGroup("lstm_2"),
            lstm_2_U_c = try layer2.openFloatDataset("lstm_2_U_c")?.read(),
            lstm_2_U_f = try layer2.openFloatDataset("lstm_2_U_f")?.read(),
            lstm_2_U_i = try layer2.openFloatDataset("lstm_2_U_i")?.read(),
            lstm_2_U_o = try layer2.openFloatDataset("lstm_2_U_o")?.read(),
            lstm_2_W_c = try layer2.openFloatDataset("lstm_2_W_c")?.read(),
            lstm_2_W_f = try layer2.openFloatDataset("lstm_2_W_f")?.read(),
            lstm_2_W_i = try layer2.openFloatDataset("lstm_2_W_i")?.read(),
            lstm_2_W_o = try layer2.openFloatDataset("lstm_2_W_o")?.read(),
            lstm_2_b_c = try layer2.openFloatDataset("lstm_2_b_c")?.read(),
            lstm_2_b_f = try layer2.openFloatDataset("lstm_2_b_f")?.read(),
            lstm_2_b_i = try layer2.openFloatDataset("lstm_2_b_i")?.read(),
            lstm_2_b_o = try layer2.openFloatDataset("lstm_2_b_o")?.read()
            else {
                return
        }
        
        let layer2_weights = Poet.makeWeightsFromComponents(
            Wc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_c),
            Wf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_f),
            Wi: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_i),
            Wo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_W_o),
            Uc: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_c),
            Uf: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_f),
            Ui: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_i),
            Uo: Matrix(rows: unitCount, columns: unitCount, elements: lstm_2_U_o))
        
        let lstm_2 = LSTMLayer(
            weights: layer2_weights,
            biases: ValueArray([ lstm_2_b_i, lstm_2_b_c, lstm_2_b_f, lstm_2_b_o ].flatMap({ $0 })),
            batchSize: batchSize,
            name: "lstm_2")
        
        guard let
            denseLayer = file.openGroup("dense_1"),
            dense_1_W = try denseLayer.openFloatDataset("dense_1_W")?.read(),
            dense_1_b = try denseLayer.openFloatDataset("dense_1_b")?.read()
        else {
            return
        }

        let dense_1 = InnerProductLayer(weights: Matrix(rows: unitCount, columns: inputSize, elements: dense_1_W),
                                        biases: ValueArray(dense_1_b))

        let input = Matrix<Float>(rows: 1, columns: inputSize)
        for i in 0..<inputSize {
            // Input vector with '1' at the index of the character we will pass in
            input[0, i] = (chars[i] == "e") ? 1 : 0
        }
        
        let dataLayer = Source(name: "input", data: input.elements, batchSize: batchSize)
        let sink = Sink(name: "sink", inputSize: inputSize, batchSize: 1)
        
        let net = Net.build {
            dataLayer => lstm_1 => lstm_2 => dense_1 => sink
        }
        
        let evaluator = try Evaluator(net: net, device: device)
        
        let semaphore = dispatch_semaphore_create(0)
        
        while true {
            evaluator.evaluate { (snapshot) in
                let output = [Float](snapshot.outputOfLayer(dense_1)!)
                
                let exps = output.map(expf)
                let sum  = exps.reduce(0, combine: +)
                let softmax = exps / sum
                
                var index = 0, maxValue: Float = 0, maxIndex = 0
                softmax.forEach({ (value) in
                    if value > maxValue {
                        maxIndex = index
                        maxValue = value
                    }
                    index += 1
                })
                
                print(chars[maxIndex], terminator: "")
                
                let input = Matrix<Float>(rows: 1, columns: inputSize)
                for i in 0..<inputSize {
                    // Input vector with '1' at the index of the character we will pass in
                    input[0, i] = (chars[i] == chars[maxIndex]) ? 1 : 0
                }
                
                dataLayer.data = input.elements
                
                dispatch_semaphore_signal(semaphore);
            }
            dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER)
        }
   }
    
    /// Make an LSTM weight matrix from separate W and U component matrices.
    static func makeWeightsFromComponents(Wc Wc: Matrix<Float>, Wf: Matrix<Float>, Wi: Matrix<Float>, Wo: Matrix<Float>, Uc: Matrix<Float>, Uf: Matrix<Float>, Ui: Matrix<Float>, Uo: Matrix<Float>) -> Matrix<Float> {
        let unitCount = Uc.rows
        let inputSize = Wc.rows
        
        let elements = ValueArray<Float>(count: (inputSize + unitCount) * 4 * unitCount)
        
        for i in 0..<inputSize {
            let start = i * 4 * unitCount
            elements.replaceRange(0 * unitCount + start..<0 * unitCount + start + unitCount, with: Wi.row(i))
            elements.replaceRange(1 * unitCount + start..<1 * unitCount + start + unitCount, with: Wc.row(i))
            elements.replaceRange(2 * unitCount + start..<2 * unitCount + start + unitCount, with: Wf.row(i))
            elements.replaceRange(3 * unitCount + start..<3 * unitCount + start + unitCount, with: Wo.row(i))
        }
        
        for i in 0..<unitCount {
            let start = (inputSize + i) * 4 * unitCount
            elements.replaceRange(0 * unitCount + start..<0 * unitCount + start + unitCount, with: Ui.row(i))
            elements.replaceRange(1 * unitCount + start..<1 * unitCount + start + unitCount, with: Uc.row(i))
            elements.replaceRange(2 * unitCount + start..<2 * unitCount + start + unitCount, with: Uf.row(i))
            elements.replaceRange(3 * unitCount + start..<3 * unitCount + start + unitCount, with: Uo.row(i))
        }
        
        return Matrix<Float>(rows: inputSize + unitCount, columns: 4 * unitCount, elements: elements)
    }
    
}
