//  Copyright © 2016 Alejandro Isaza. All rights reserved.

import BrainCore
import HDF5Kit
import Upsurge


class Source: DataLayer {
    let id = NSUUID()
    let name: String? = nil

    var size: Int
    var data: Blob

    var outputSize: Int {
        return size
    }

    init(size: Int) {
        data = Blob(count: size)
        self.size = size
    }

    func nextBatch(batchSize: Int) -> Blob {
        precondition(batchSize == 1)
        return data
    }
}


class Sink: SinkLayer {
    let id = NSUUID()
    let name: String? = nil

    var size: Int
    var data: Blob = []

    init(size: Int) {
        self.size = size
    }

    var inputSize: Int {
        return size
    }

    func consume(input: Blob) {
        self.data = input
    }
}


class NetworkBuilder {
    static let inputSize = 57
    static let outputSize = 57
    
    let dataLayer = Source(size: inputSize)
    let sinkLayer = Sink(size: outputSize)

    init() {

    }

    func loadNetFromFile(path: String) -> Net {
        guard let file = File.open(path, mode: .ReadOnly) else {
            fatalError("File not found '\(path)'")
        }

        let lstm_1 = try! loadLSTMLayerFromFile(file, name: "lstm_1")
        let lstm_2 = try! loadLSTMLayerFromFile(file, name: "lstm_2")
        let denseLayer = loadDenseLayerFromFile(file)

        return Net.build {
            self.dataLayer => lstm_1 => lstm_2 => denseLayer => self.sinkLayer
        }
    }

    private func loadLSTMLayerFromFile(file: File, name: String) throws -> LSTMLayer {
        guard let group = file.openGroup(name) else {
            fatalError("LSTM \(name) group not found in file")
        }

        guard let
            ucDataset = group.openFloatDataset("\(name)_U_c"),
            ufDataset = group.openFloatDataset("\(name)_U_f"),
            uiDataset = group.openFloatDataset("\(name)_U_i"),
            uoDataset = group.openFloatDataset("\(name)_U_o"),
            wcDataset = group.openFloatDataset("\(name)_W_c"),
            wfDataset = group.openFloatDataset("\(name)_W_f"),
            wiDataset = group.openFloatDataset("\(name)_W_i"),
            woDataset = group.openFloatDataset("\(name)_W_o"),
            bcDataset = group.openFloatDataset("\(name)_b_c"),
            bfDataset = group.openFloatDataset("\(name)_b_f"),
            biDataset = group.openFloatDataset("\(name)_b_i"),
            boDataset = group.openFloatDataset("\(name)_b_o")
            else {
                fatalError("LSTM weights for \(name) not found in file")
        }

        let inputSize = wcDataset.space.dims[0]
        let unitCount = wcDataset.space.dims[1]

        let weights = LSTMLayer.makeWeightsFromComponents(
            Wc: Matrix(rows: inputSize, columns: unitCount, elements: try wcDataset.read()),
            Wf: Matrix(rows: inputSize, columns: unitCount, elements: try wfDataset.read()),
            Wi: Matrix(rows: inputSize, columns: unitCount, elements: try wiDataset.read()),
            Wo: Matrix(rows: inputSize, columns: unitCount, elements: try woDataset.read()),
            Uc: Matrix(rows: unitCount, columns: unitCount, elements: try ucDataset.read()),
            Uf: Matrix(rows: unitCount, columns: unitCount, elements: try ufDataset.read()),
            Ui: Matrix(rows: unitCount, columns: unitCount, elements: try uiDataset.read()),
            Uo: Matrix(rows: unitCount, columns: unitCount, elements: try uoDataset.read()))

        let biases = ValueArray([
            try biDataset.read(),
            try bcDataset.read(),
            try bfDataset.read(),
            try boDataset.read()
            ].flatMap({ $0 }))

        return LSTMLayer(weights: weights, biases: biases, batchSize: 1, name: name)
    }

    private func loadDenseLayerFromFile(file: File) -> InnerProductLayer {
        guard let group = file.openGroup("dense_1") else {
            fatalError("Dense group not found in file")
        }

        guard let weightsDataset = group.openFloatDataset("dense_1_W"), weights = try? weightsDataset.read() else {
            fatalError("Dense weights not found in file")
        }

        guard let biasesDataset = group.openFloatDataset("dense_1_b"), biases = try? biasesDataset.read() else {
            fatalError("Dense biases not found in file")
        }

        let inputSize = weightsDataset.space.dims[0]
        let outputSize = weightsDataset.space.dims[1]
        let weightsMatrix = Matrix(rows: inputSize, columns: outputSize, elements: weights)
        return InnerProductLayer(weights: weightsMatrix, biases: ValueArray(biases), name: "dense_1")
    }
}
