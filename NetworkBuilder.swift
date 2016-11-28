//  Copyright © 2016 Alejandro Isaza. All rights reserved.

import BrainCore
import HDF5Kit
import Upsurge


class Source: DataLayer {
    /// Unique layer identifier
    public var id: UUID = UUID()

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

    func nextBatch(_ batchSize: Int) -> Blob {
        precondition(batchSize == 1)
        return data
    }
}


class Sink: SinkLayer {
    /// Unique layer identifier
    public var id: UUID = UUID()

    let name: String? = nil

    var size: Int
    var data: Blob = []

    init(size: Int) {
        self.size = size
    }

    var inputSize: Int {
        return size
    }

    func consume(_ input: Blob) {
        self.data = input
    }
}


class NetworkBuilder {
    var dataLayer: Source
    var sinkLayer: Sink

    init(inputSize: Int, outputSize: Int) {
        dataLayer = Source(size: inputSize)
        sinkLayer = Sink(size: outputSize)
    }

    func loadNetFromFile(_ path: String) -> Net {
        guard let file = File.open(path, mode: .readOnly) else {
            fatalError("File not found '\(path)'")
        }

        let lstm_1 = try! loadLSTMLayerFromFile(file, name: "lstm_1")
        let lstm_2 = try! loadLSTMLayerFromFile(file, name: "lstm_2")
        let denseLayer = loadDenseLayerFromFile(file)

        return Net.build {
            self.dataLayer => lstm_1 => lstm_2 => denseLayer => self.sinkLayer
        }
    }

    fileprivate func loadLSTMLayerFromFile(_ file: File, name: String) throws -> LSTMLayer {
        guard let group = file.openGroup(name) else {
            fatalError("LSTM \(name) group not found in file")
        }

        guard let
            ucDataset = group.openFloatDataset("\(name)_U_c"),
            let ufDataset = group.openFloatDataset("\(name)_U_f"),
            let uiDataset = group.openFloatDataset("\(name)_U_i"),
            let uoDataset = group.openFloatDataset("\(name)_U_o"),
            let wcDataset = group.openFloatDataset("\(name)_W_c"),
            let wfDataset = group.openFloatDataset("\(name)_W_f"),
            let wiDataset = group.openFloatDataset("\(name)_W_i"),
            let woDataset = group.openFloatDataset("\(name)_W_o"),
            let bcDataset = group.openFloatDataset("\(name)_b_c"),
            let bfDataset = group.openFloatDataset("\(name)_b_f"),
            let biDataset = group.openFloatDataset("\(name)_b_i"),
            let boDataset = group.openFloatDataset("\(name)_b_o")
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

    fileprivate func loadDenseLayerFromFile(_ file: File) -> InnerProductLayer {
        guard let group = file.openGroup("dense_1") else {
            fatalError("Dense group not found in file")
        }

        guard let weightsDataset = group.openFloatDataset("dense_1_W"), let weights = try? weightsDataset.read() else {
            fatalError("Dense weights not found in file")
        }

        guard let biasesDataset = group.openFloatDataset("dense_1_b"), let biases = try? biasesDataset.read() else {
            fatalError("Dense biases not found in file")
        }

        let inputSize = weightsDataset.space.dims[0]
        let outputSize = weightsDataset.space.dims[1]
        let weightsMatrix = Matrix(rows: inputSize, columns: outputSize, elements: weights)
        return InnerProductLayer(weights: weightsMatrix, biases: ValueArray(biases), name: "dense_1")
    }
}
