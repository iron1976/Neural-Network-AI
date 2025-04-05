#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <thread> 
#include <Windows.h> //Windows.h for key inputs
#include <conio.h>
#include <limits>
#include <list> 
#include <numeric> 
namespace NeuralNetwork
{
    // WORK ON: BIASES ALSO CHANGE WITH BACKPROPAGATION
    namespace Settings
    {
        const double LearningRate = (double)0.1;
        const std::vector<std::vector<double>> Input = { {0}, {1} }; 
        const std::vector<std::vector<double>> ExpectedOutput = { {0}, {1} }; 
        const std::vector<unsigned int> HiddenLayerSize = { 2 };

        //Same size with Start Weights: TotalLayers - 1
        const std::vector<double> BiasData = { 0.5, 0.5 };

        //Size of std::vector<std::vector<std::vector<double>>> -> TotalLayers - 1
        //Size of std::vector<std::vector<double>> -> TotalNeurons
        //Size of std::vector<double> -> TotalWeights 
        std::vector<std::vector<std::vector<double>>> StartWeights = std::vector<std::vector<std::vector<double>>>{

            { //Input-Hidden
                {0.5, }, {-0.5 }
            },


            { //Hidden-Output 
                { 0.5, 0.5 },
            }

        };

        //TRUE DATA
        //std::vector<std::vector<std::vector<double>>> StartWeights = std::vector<std::vector<std::vector<double>>>{
        //
        //{ {1,1      },{1,1     },{1,1     }},//Input-Hidden
        //
        //{ {1,1,1      },{1,1 ,1    } },//Input-Hidden
        //
        //{ {1,1       }  },//Input-Hidden
        //
        //{ {1 },    }//Hidden-Output 
        //
        //};

        /// <summary>
        /// If starting weight is missing in StartWeights. Set weight to StartWeightsDefault.
        /// </summary>
        const double StartWeightsDefault = 1.0;
        const bool ShowNetIDsInShowWeightMode = true; 
        enum KeyboardKeys : char
        {
            FeedForward = 'w',
            ShowNetValues = 'a',
            ShowNetWeights = 's',
            ShowNetID = 'd',
            EnterLayer = 'v',
            EnterEpochsTarget = 'b',
            EmptyContinue = 32,//Spacebar
            DebugTestKey = 'g',

        };
    }
    using namespace Settings;

    #pragma region ...
    
    namespace Math
    {
        //sigmoid function f(x) = 1/(1 + e^-x)
        auto ActivationFormula = [&](double x) {
            return 1.0 / (1.0 + exp(-x));
        };

        const double delta = 0.0000001;
        auto derivative = [](auto foo) {
            return [&](double x) {
                return (foo(x + delta) - foo(x)) / delta;
            };
        };
        auto derivative2 = [](auto foo) {
            return [&](double x, double y) {
                return (foo(x + delta, y) - foo(x, y)) / delta;
            };
        };
        auto derivative3 = [](auto foo) {
            return [&](double x, double y, double z) {
                return (foo(x + delta, y, z) - foo(x, y, z)) / delta;
            };
        };
    } 

     
    const unsigned int InputLayerSize = Input[0].size();
    const unsigned int OutputLayerSize = ExpectedOutput[0].size();
    const unsigned int TotalLayers = 2 + HiddenLayerSize.size();

    namespace Screen
    {
        enum LayerTypeEnum : char
        {
            NonLayer = 0,
            InputLayer = 1,
            HiddenLayer = 2,
            OutputLayer = 3
        };
        enum OutputTypeEnum : char
        {
            null = 0,
            ShowNetValue = 1,
            ShowNetWeights = 2,
            ShowNetID = 3,
            Length = 4,
        };
        static bool IsScreenSet = false;
        std::string Output = "";

        void print(std::string logs)
        {
            Screen::Output += logs + "\n";
        }
        std::string KeyGuide;
        std::string CurrentMode;
        std::string ModeSelection;
        std::string StaticLogs;
        class ScreenNets;
        static std::vector<ScreenNets*> ScreenNetsAllList;
        static std::vector<ScreenNets*> ScreenNetsInputList = std::vector<ScreenNets*>(InputLayerSize);
        static std::vector<std::vector<ScreenNets*>> ScreenNetsHiddenList = ([&]() {
            std::vector<std::vector<ScreenNets*>> result = std::vector<std::vector<ScreenNets*>>(HiddenLayerSize.size());
            for (unsigned int j = 0; j < result.size(); j++)
                result[j] = std::vector<ScreenNets*>(HiddenLayerSize[j]);

            return result;
            })();
        static std::vector<ScreenNets*> ScreenNetsOutputList = std::vector<ScreenNets*>(OutputLayerSize);

        class ScreenNets

        {

        public:
            unsigned int BaseIndex;
            std::string NetName;
            double Value = 0;
            std::vector<double>* WeightValues;
            unsigned int SelectedWeightIndex = 0;
            OutputTypeEnum OutputType = OutputTypeEnum::ShowNetID;
            std::string NetID;

            ScreenNets(unsigned int BaseIndex, LayerTypeEnum LayerType, std::string NetID, unsigned int Index1, unsigned int Index2) : BaseIndex(BaseIndex), NetID(NetID), NetName("BaseIndex: " + std::to_string(BaseIndex))
            {
                if (NetID.length() != 4)
                    throw std::invalid_argument("Net ID length isn't 4");
                ScreenNetsAllList.push_back(this);
                if (LayerType == LayerTypeEnum::InputLayer)
                {
                    //std::cout << "\n INPUT LAYER ADDED ";
                    ScreenNetsInputList[Index1] = (this);
                }
                else if (LayerType == LayerTypeEnum::HiddenLayer)
                {
                    //std::cout << "\n HIDDEN LAYER ADDED ";
                    ScreenNetsHiddenList[Index2][Index1] = (this);
                }
                else if (LayerType == LayerTypeEnum::OutputLayer)
                {
                    //std::cout << "\n OUTPUT LAYER ADDED ";
                    ScreenNetsOutputList[Index1] = (this);
                }
            }
            void SetNet(double Value)
            {
                if (!IsScreenSet)
                    throw std::invalid_argument("error");
                if (this == nullptr)
                    throw std::invalid_argument("This net doesn't exist maybe out of bounds.");

                this->Value = Value;
                UpdateScreenOutput();
            }
            void SetWeights(std::vector<double>* weightValues)
                {
                    if (!IsScreenSet)
                        throw std::invalid_argument("error");
                    if (this == nullptr)
                        throw std::invalid_argument("This net doesn't exist maybe out of bounds.");

                    this->WeightValues = weightValues;
                    UpdateScreenOutput();

                }
            void SelectWeight(unsigned int index)
                {
                    this->SelectedWeightIndex = index;
                }
            void SetOutputType(OutputTypeEnum outputType)
                {
                    if (this == nullptr)
                        throw std::invalid_argument("This net doesn't exist maybe out of bounds.");
                    this->OutputType = outputType;
                    //std::cout << "SetOutputType";
                    UpdateScreenOutput();
                }
            void UpdateScreenOutput() 
                {
                    if (this == nullptr)
                        throw std::invalid_argument("This net doesn't exist maybe out of bounds.");
                    if (OutputType == OutputTypeEnum::ShowNetValue)
                    {
                        std::string valueShorted = "     ";
                        if (this->Value >= 1)
                            valueShorted = std::to_string(Value).substr(0, 4);
                        else
                            valueShorted = std::to_string(Value).substr(1, 4);

                        std::string somestr = "| " + valueShorted + " |     ";
                        Screen::Output = Screen::Output.substr(0, BaseIndex - 1) + somestr + Screen::Output.substr(BaseIndex - 1 + somestr.size());
                    }
                    else if (OutputType == OutputTypeEnum::ShowNetWeights)
                    {
                        if (this->WeightValues == nullptr)
                            throw std::invalid_argument("WeightValues isn't set");
                        double value = (*this->WeightValues)[SelectedWeightIndex];
                        std::string valueShorted = "     ";
                        if (value >= 1)
                            valueShorted = std::to_string(value).substr(0, 4);
                        else
                            valueShorted = std::to_string(value).substr(1, 4);

                        std::string somestr = "| " + valueShorted + " |     ";
                        Screen::Output = Screen::Output.substr(0, BaseIndex - 1) + somestr + Screen::Output.substr(BaseIndex - 1 + somestr.size());
                    }
                    else if (OutputType == OutputTypeEnum::ShowNetID)
                    {
                        std::string somestr = "| " + NetID + " |     ";
                        Screen::Output = Screen::Output.substr(0, BaseIndex - 1) + somestr + Screen::Output.substr(BaseIndex - 1 + somestr.size());
                    }
                }
        };

        void ClearScreen()
            {
                system("cls");
                for (int i = 0; i < 30; i++) { std::cout << "\x0A"; }
                std::cout << "\x1b[H";
                return;
                for (unsigned int j = 0; j < 1500; j++)
                {
                    std::cout << "\033[F";
                }
            }
        void PrintNeurons()
        {

            std::cout << Output << "\033[F" << "\033[F";
        }
        void PrintLogs()
        { 
            std::cout << "\033[94m" + KeyGuide + "\033[0m" << "\n";//BRIGHT_BLUE
            if (CurrentMode.size() > 0)
                std::cout << "\033[94m" + CurrentMode + "\033[0m" << "\n";//BRIGHT_BLUE
            if (StaticLogs.size() > 0)
                std::cout << "\033[92m" + StaticLogs + "\033[0m" << "\n";//BRIGHT_GREEN
            if (ModeSelection.size() > 0)
                std::cout << "\033[31m" + ModeSelection + "\033[0m" << "\n";//DARK_RED
        }
        void PrintScreen()
        {
            //For colors: https://ss64.com/nt/syntax-ansi.html
            ClearScreen();
            PrintNeurons();
            PrintLogs();

        };
        void SetScreen()
        {
            //Also Input Size

            //Setting Bias Screen


            //Setting Net Screen

            unsigned int BlockRowLength = 0;


            for (unsigned int j = 0; j < HiddenLayerSize.size(); j++)
            {
                unsigned int val = HiddenLayerSize[j];
                if (BlockRowLength < val)
                    BlockRowLength = val;
            }

            if (InputLayerSize > BlockRowLength)
                BlockRowLength = InputLayerSize;
            
            if (OutputLayerSize > BlockRowLength)
                BlockRowLength = BlockRowLength;
            

            unsigned int RowSize = BlockRowLength * 4; 

            std::string fullOutput = "";
            unsigned int InputIndex = 0;
            std::vector<unsigned int> HiddenIndex = std::vector<unsigned int>(HiddenLayerSize.size());
            unsigned int OutputIndex = 0;

            unsigned int InputStartIndex = (int)BlockRowLength - (int)InputLayerSize < 0 ? 0 : BlockRowLength - InputLayerSize;

            std::vector<unsigned int> HiddenStartIndex = std::vector<unsigned int>(HiddenLayerSize.size());
            for (unsigned int j = 0; j < HiddenStartIndex.size(); j++)
                HiddenStartIndex[j] = (int)InputLayerSize - (int)HiddenLayerSize[j] < 0 ? 0 : BlockRowLength - HiddenLayerSize[j];

            unsigned int OutputStartIndex = (int)BlockRowLength - (int)OutputLayerSize < 0 ? 0 : BlockRowLength - OutputLayerSize;


            std::vector<bool> HiddenPair = std::vector<bool>(HiddenLayerSize.size());
            for (unsigned int j = 0; j < HiddenPair.size(); j++)
                HiddenPair[j] = HiddenStartIndex[j] % 2;

            bool OutputPair = BlockRowLength % 2 ? !(OutputLayerSize % 2) : (OutputLayerSize % 2);
            bool InputPair = BlockRowLength % 2 ? !(InputLayerSize % 2) : (InputLayerSize % 2);

             
            for (unsigned int i = 0, TBaseIndex = 0; i < (BlockRowLength * 2)-1; i++)
            {
                std::string oneRow = "";

                bool DrawInputNet = false;
                {
                    if (i >= InputStartIndex && InputIndex < InputLayerSize)
                    {
                        if (i % 2 == InputPair)
                        {
                            DrawInputNet = true;
                            InputIndex++;
                        }
                        else
                            DrawInputNet = false;
                    }
                }
                std::vector<bool> DrawHiddenNet = std::vector<bool>(HiddenLayerSize.size());
                {
                    for (unsigned int c = 0; c < HiddenStartIndex.size(); c++)
                        if (i >= HiddenStartIndex[c] && HiddenIndex[c] < HiddenLayerSize[c])
                        {

                            if (i % 2 == HiddenPair[c]  )
                            {
                                DrawHiddenNet[c] = true;
                                HiddenIndex[c]++;
                            }

                        }
                } 
                bool DrawOutputNet = false;
                {
                    if (i >= OutputStartIndex && OutputIndex < OutputLayerSize)
                    {
                        if (i % 2 == OutputPair)
                        {
                            DrawOutputNet = true;
                            OutputIndex++;
                        }
                        else
                            DrawOutputNet = false;
                    }
                }
                 
                //Drawing Net
                  
                unsigned int IndexForNet = 0;
                oneRow = "      ";
                for (unsigned int k = 0; k < TotalLayers; k++)
                    if ((DrawInputNet && k == 0) ||
                        (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1) ||
                        (k != 0 && DrawHiddenNet[k - 1]))
                        oneRow += "\033[94m+------+  \033[0m    ";
                    else
                        oneRow += "              ";

                oneRow += "\n";

                oneRow += "      ";
                for (unsigned int k = 0; k < TotalLayers; k++)
                {

                    LayerTypeEnum LayerType = LayerTypeEnum::NonLayer; 
                    unsigned int Value = -1;

                    if ((DrawInputNet && k == 0))
                    {
                        LayerType = LayerTypeEnum::InputLayer;
                        Value = InputIndex;
                    }
                    else if (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1)
                    {
                        LayerType = LayerTypeEnum::OutputLayer;
                        Value = OutputIndex;
                    }
                    else if (k != 0 && DrawHiddenNet[k - 1])
                    {
                        LayerType = LayerTypeEnum::HiddenLayer;
                        Value = HiddenIndex[k - 1];
                    } 
                    
                    bool kIs2Digits = true;
                    if (k < 10)
                        kIs2Digits = false;

                    if (Value != (unsigned int)-1)
                    {
                        if (Value - 1 < 10)
                        {
                            if (kIs2Digits)
                                oneRow += "|\033[35m" + (std::string)std::to_string(k) + "_" + (std::string)std::to_string(Value - 1) + "\033[0m |      ";//Total Length: 15
                            else
                                oneRow += "| " + (std::string)std::to_string(k) + "__" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                        }
                        else
                        {
                            if (kIs2Digits)
                                oneRow += "| " + (std::string)std::to_string(k) + "" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                            else
                                oneRow += "| " + (std::string)std::to_string(k) + "_" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                        }
                        if (Output.size() + oneRow.size() - 13 < 0)
                            throw std::invalid_argument("error");
                        TBaseIndex = (Output.size() + oneRow.size() - 13);

                        new ScreenNets(TBaseIndex, LayerType, oneRow.substr(oneRow.length() - 12, 4), Value - 1, k - 1);
                    }
                    else
                        oneRow += "              ";
                }
                oneRow += "\n";
                oneRow += "      ";
                for (unsigned int k = 0; k < TotalLayers; k++)
                    if ((DrawInputNet && k == 0) ||
                        (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1) ||
                        (k != 0 && DrawHiddenNet[k - 1]))
                        oneRow += "\033[35m+------+  \033[0m    ";
                    else
                        oneRow += "              ";

                oneRow += "\n";
                
                Output += oneRow;
            }
            Output += "       ";
            for (int c = 0; c < 1; c++)
            { 

                for (unsigned int j = 0; j < 1; j++)
                {
                    for(unsigned int i = 0; i < TotalLayers-1; i++)
                        Output += "      \033[35m*-BIAS-*\033[0m";
                    Output += "\n";
                    Output += "       ";

                    for (unsigned int i = 0; i < TotalLayers - 1; i++)
                    {
                        double value = BiasData[i];
                        bool Is2Digits = true;
                        if (value < 1)
                            Is2Digits = false;
                        if (Is2Digits)
                            Output += "      |\033[35m " + std::to_string(value).substr(0, 4) + "\033[0m |"; 
                        else
                            Output += "      |\033[35m " + std::to_string(value).substr(1, 4) + "\033[0m |";
                    }  
                }
            }
            Output += "\n\n\n";
             
            IsScreenSet = true;
        }

    }
    using namespace Screen;
    namespace NeuralNeurons
    {

        class Neuron;
         
        static std::vector<Neuron*> AllLayerNeurons = std::vector<Neuron*>();
        static std::vector<Neuron*> InputLayerNeurons = std::vector<Neuron*>(InputLayerSize);
        static std::vector<std::vector<Neuron*>> HiddenLayerNeurons = ([&]() {
            std::vector<std::vector<Neuron*>> result = std::vector<std::vector<Neuron*>>(HiddenLayerSize.size());
            for (unsigned int j = 0; j < result.size(); j++)
                result[j] = std::vector<Neuron*>(HiddenLayerSize[j]);

            return result;
        })();
        static std::vector<Neuron*> OutputLayerNeurons = std::vector<Neuron*>(OutputLayerSize);


        static double Cost;
        static double CostPerEpochs;
        static double CostPerEpochsAverage;
        static std::vector<double> CostPerEpochsList;
        static std::vector<double> CostPerEpochsAverageList;

        unsigned int BatchIndex = 0;
        unsigned int EpochsIndex = 0;

        class Neuron
        {
        public:

            Screen::ScreenNets* ScreenNet;

            LayerTypeEnum LayerType;
            /// <summary>
                /// Vertical index of Neuron.
                /// </summary>
                /// <param name="ScreenNet"></param>
                /// <param name="ConnectedNeurons"></param>
                /// <param name="Weights"></param>
                /// <param name="LayerType"></param>
            unsigned short NeuronIndex;
            unsigned short NeuronLayerIndex;


            /// <summary>
                /// Data going to be added from right side neurons.
                /// </summary>
            std::vector<double> OutputWeights;
            double Output = 0;
            double ActivationOutput;
            std::vector<Neuron*> ConnectedNeurons;
            bool IsValid;


            bool IsOutputSet;
            bool IsInputNeuron;

            /// <summary>
                /// Setting hidden layer neurons or output neuron.
                /// </summary>
                /// <param name="ConnectedNeurons"></param>
                /// <param name="Weights"></param>
                /// <param name="NeuronLength"></param>
            Neuron(Screen::ScreenNets* ScreenNet, std::vector<Neuron*> ConnectedNeurons, std::vector<double> Weights, LayerTypeEnum LayerType, unsigned short NeuronIndex, unsigned short NeuronLayerIndex = 0)
            {
                if (ConnectedNeurons.size() != Weights.size())
                {
                    std::cout << "Sizes don't match: " << ConnectedNeurons.size() << " " << Weights.size();
                    throw std::invalid_argument("Sizes don't match.");
                }
                this->ConnectedNeurons = ConnectedNeurons;
                this->NeuronIndex = NeuronIndex;
                this->NeuronLayerIndex = NeuronLayerIndex;
                this->ScreenNet = ScreenNet;
                this->IsInputNeuron = false;
                this->IsOutputSet = false;
                this->LayerType = LayerType;
                this->IsValid = true;
                this->OutputWeights = std::vector<double>();
                InitializeWeights(&Weights);
                SetWeights(&Weights);
            }

            /// <summary>
                /// Setting Input OR BIAS.
                /// </summary>
                /// <param name="Current"></param>
            Neuron(Screen::ScreenNets* ScreenNet, double Output, LayerTypeEnum LayerType, unsigned short NeuronIndex) : ScreenNet(ScreenNet), Output(Output), LayerType(LayerType), IsInputNeuron(true), IsOutputSet(true), NeuronIndex(NeuronIndex), NeuronLayerIndex(0), IsValid(true) {

                    this->OutputWeights = std::vector<double>();
                    SetOutput();
                }
        private:
            void SumFunction()
            {
                Output = 0;

                for (unsigned int j = 0; j < ConnectedNeurons.size(); j++)
                {
                    Neuron* neuron = ConnectedNeurons[j];
                   //if (neuron->IsInputNeuron)
                   //{
                   //    Output += neuron->Output * neuron->OutputWeights[this->NeuronIndex];
                   //
                   //}
                   //else
                    {

                        double outputWeight = neuron->OutputWeights[this->NeuronIndex];
                        if (isinf(outputWeight))
                        {
                            if (outputWeight > 0)//Positive infinity
                                Output += neuron->ActivationOutput * 1;
                            else//Negative infinity
                                Output += neuron->ActivationOutput * 0;
                            std::cout << outputWeight;
                            throw std::invalid_argument("INFINITY FOUND AGAIN");
                        }

                        Output += neuron->ActivationOutput * outputWeight; 
                    }
                }

                if (isnan(Output))
                {
                    std::cout << "nan value found: " << Output;
                    throw std::invalid_argument("Nan value found!");
                }
                if (this->LayerType == LayerTypeEnum::HiddenLayer)
                    Output += BiasData[NeuronLayerIndex];
                else if (this->LayerType == LayerTypeEnum::OutputLayer)
                    Output += BiasData[BiasData.size() - 1];

                if (isnan(Output))
                {
                    std::cout << "nan value found: " << Output;
                    throw std::invalid_argument("Nan value found!");
                }
            }
            void ActivationFunction()
            {
                ActivationOutput = Math::ActivationFormula(Output);
                if (isnan(ActivationOutput))
                {
                    std::cout << "nan value found: " << Output;
                    throw std::invalid_argument("Nan value found!");
                }
            
            }
            void InitializeWeights(const std::vector<double>* Weights)
            {
                for (unsigned int j = 0; j < Weights->size(); j++)
                {
                    this->ConnectedNeurons[j]->OutputWeights.push_back((*Weights)[j]);
                }
            }
        public:
            void SetWeights(std::vector<double>* Weights)
            {
                for (unsigned int j = 0; j < Weights->size(); j++)
                    SetWeightAt(j, (*Weights)[j]);
            }
            void SetWeightAt(unsigned int j, double Value)
                {
                    this->ConnectedNeurons[j]->OutputWeights[this->NeuronIndex] = Value;
                    this->ConnectedNeurons[j]->ScreenNet->SetWeights(&this->ConnectedNeurons[j]->OutputWeights);
                }
            void SetShowWeights()
            {
                for (unsigned int j = 0; j < this->ConnectedNeurons.size(); j++)
                    this->ConnectedNeurons[j]->ScreenNet->SelectWeight(this->NeuronIndex);
            }
            void SetOutputValueForInputNeuron(double Value)
            {
                if (!IsInputNeuron)
                    throw std::invalid_argument("This isn't Input neuron");
            
                Output = Value;
                SetOutput();
            }
            void SetOutput()
            {
                if (IsInputNeuron)
                {
                    ScreenNet->SetNet(Output);
            
                }
                else
                {
                    IsOutputSet = false;
                    SumFunction();
                    ActivationFunction();
                    IsOutputSet = true;
                    ScreenNet->SetNet(ActivationOutput);
                }
            }
        };

        std::vector<double> CostDifference = std::vector<double>(ExpectedOutput.size());

        void InitializeNeurons()
        {  
            std::vector<std::vector<std::vector<double>>> OldStartWeights = StartWeights; 

            bool isFixedStartWeightsUsed = false;

            if (ScreenNetsInputList.size() != InputLayerSize)
            {
                std::cout << "\n\n ERROR: ScreenNetsInput And InputLayerSize isn't equal: " + std::to_string(ScreenNetsInputList.size()) + " " + std::to_string(InputLayerSize);
                throw std::invalid_argument("ScreenNetsInput And InputLayerSize isn't equal: " + std::to_string(ScreenNetsInputList.size()) + " " + std::to_string(InputLayerSize));
            }
            for (unsigned int j = 0; j < ScreenNetsHiddenList.size(); j++)
                if (ScreenNetsHiddenList[j].size() != HiddenLayerSize[j])
                {
                    std::cout << "\n\n ERROR: ScreenNetsHidden And HiddenLayerSize isn't equal: " + std::to_string(ScreenNetsHiddenList[j].size()) + " " + std::to_string(HiddenLayerSize[j]);
                    throw std::invalid_argument("ScreenNetsHidden And HiddenLayerSize isn't equal: " + std::to_string(ScreenNetsHiddenList[j].size()) + " " + std::to_string(HiddenLayerSize[j]));
                }
            if (ScreenNetsOutputList.size() != OutputLayerSize)
            {
                std::cout << "\n\n ERROR: ScreenNetsOutput And OutputLayerSize isn't equal: " + std::to_string(ScreenNetsOutputList.size()) + " " + std::to_string(OutputLayerSize);
                throw std::invalid_argument("ScreenNetsOutput And OutputLayerSize isn't equal: " + std::to_string(ScreenNetsOutputList.size()) + " " + std::to_string(OutputLayerSize));
            }
             
            if (InputLayerSize != ScreenNetsInputList.size())
                throw std::invalid_argument("Sizes don't match."); 
            
            if (StartWeights.size() != TotalLayers - 1)//Checking Total Layers
            {
                std::cout << StartWeights.size() << " " << TotalLayers;
                std::cout << "WARNING(0): Start Weight size has missing data. Setting automatically.\n";
                isFixedStartWeightsUsed = true;
                throw std::invalid_argument("Sizes don't match.");
                StartWeights = std::vector<std::vector<std::vector<double>>>(TotalLayers - 1);
            } 
             
            if (StartWeights[0].size() != HiddenLayerSize[0])//Checking Input Layer 1
            {
                std::cout << "WARNING(1): Input Layer has missing data. Setting automatically.\n";
                isFixedStartWeightsUsed = true;
                StartWeights[0] = std::vector<std::vector<double>>(HiddenLayerSize[0]);
                for (unsigned int j = 0; j < StartWeights[0].size(); j++)
                {
                    StartWeights[0][j] = std::vector<double>(InputLayerSize);
                    for (unsigned int i = 0; i < InputLayerSize; i++)
                        StartWeights[0][j][i] = StartWeightsDefault;
                }
            } 
            for(unsigned int j = 0; j < StartWeights[0].size();j++)//Checking Input Layer 2
                if (StartWeights[0][j].size() != InputLayerSize)
                {

                    std::cout << "WARNING(2): Hidden Layer has missing data. Setting automatically." << StartWeights[0][j].size() << " " << InputLayerSize << "\n";
                    isFixedStartWeightsUsed = true;
                    for (unsigned int j = 0; j < StartWeights[0].size(); j++)
                    {
                        StartWeights[0][j] = std::vector<double>(InputLayerSize);
                        for (unsigned int i = 0; i < InputLayerSize; i++)
                            StartWeights[0][j][i] = StartWeightsDefault;
                    }
                }          
            
            
            for (unsigned int j = 1; j < StartWeights.size() - 1; j++)//Checking Hidden Layer 1&2
            {
                unsigned int prevLayerSize = 0;
                bool isHiddenConnectedToInput = false;
                bool isSetNewWeights = false;
                if ((long)j - 2 < 0)
                {
                    prevLayerSize = InputLayerSize;
                    isHiddenConnectedToInput = true;
                }
                else 
                    prevLayerSize = HiddenLayerSize[j - 2]; 
                 
            
                
                if (StartWeights[j].size() != HiddenLayerSize[j]) 
                    isSetNewWeights = true; 
                else 
                    for (unsigned int k = 0; k < StartWeights[j].size(); k++) 
                        if (StartWeights[j][k].size() != HiddenLayerSize[j-1])
                            isSetNewWeights = true; 
            
                if (isSetNewWeights)
                {
                    std::cout << StartWeights[j].size() << " " << HiddenLayerSize[j];
                    std::cout << "WARNING(3): Hidden Layer Index: " << j << " has missing data. Setting automatically.\n";
                    isFixedStartWeightsUsed = true;

                    StartWeights[j] = std::vector<std::vector<double>>(HiddenLayerSize[j]);
                    for (unsigned int i = 0; i < StartWeights[j].size(); i++)
                    {
                        StartWeights[j][i] = std::vector<double>(HiddenLayerSize[j - 1]);
                        for (unsigned int k = 0; k < StartWeights[j][i].size(); k++)
                            StartWeights[j][i][k] = StartWeightsDefault;
                    }
                }
            
            } 
            
            if (StartWeights[StartWeights.size()-1].size() != OutputLayerSize)//Checking Output Layer 1
            {
                std::cout << "WARNING(4,1): Output Layer has missing data."<< StartWeights[StartWeights.size() - 1].size() << " " << OutputLayerSize << " Setting automatically.\n";
                isFixedStartWeightsUsed = true;
                StartWeights[StartWeights.size() - 1] = std::vector<std::vector<double>>(OutputLayerSize);
                for (unsigned int k = 0; k < StartWeights[StartWeights.size() - 1].size(); k++)
                {
                    StartWeights[StartWeights.size() - 1][k] = std::vector<double>(HiddenLayerSize[HiddenLayerSize.size() - 1]);
                    for (unsigned int i = 0; i < HiddenLayerSize[HiddenLayerSize.size() - 1]; i++)
                        StartWeights[StartWeights.size() - 1][k][i] = StartWeightsDefault;
                }
            }
            
            for (unsigned int j = 0; j < StartWeights[StartWeights.size() - 1].size(); j++)//Checking Output Layer 2
                if (StartWeights[StartWeights.size()-1][j].size() != HiddenLayerSize[HiddenLayerSize.size() - 1])
                { 
                    std::cout << "WARNING(4,2): Output Layer has missing data. Setting automatically.\n";
                    isFixedStartWeightsUsed = true;
                    StartWeights[StartWeights.size() - 1] = std::vector<std::vector<double>>(OutputLayerSize);
                    for (unsigned int k = 0; k < StartWeights[StartWeights.size() - 1].size(); k++)
                    {
                        StartWeights[StartWeights.size() - 1][k] = std::vector<double>(HiddenLayerSize[HiddenLayerSize.size() - 1]);
                        for (unsigned int i = 0; i < HiddenLayerSize[HiddenLayerSize.size() - 1]; i++)
                            StartWeights[StartWeights.size() - 1][k][i] = StartWeightsDefault;
                    }
                }
            
            if (BiasData.size() != TotalLayers-1)
            {
                throw std::invalid_argument("BiasData Sizes don't match with Total Layers.");
            }

            auto printAllWeights = [&](const std::vector<std::vector<std::vector<double>>>& Weight) {
             
                for (unsigned int j = 0; j < Weight.size(); j++)
                {
                    std::cout << "\n";
                    if (j == 0)
                        std::cout << "Input Layer: ";
                    else if (j == Weight.size() - 1)
                        std::cout << "Output Layer: ";
                    else
                        std::cout << "Hidden Layer(" + std::to_string(j) + "): ";
                    for (unsigned int i = 0; i < Weight[j].size(); i++)
                    {
                        std::cout << "{ ";
                        for (unsigned int k = 0; k < Weight[j][i].size(); k++)
                        {
                            if (k != 0)
                                std::cout << ", ";
                            std::cout << Weight[j][i][k];
                        }
                        std::cout << " }";
                        if (i < Weight[j].size() - 1)
                            std::cout << ", ";
                        else
                            std::cout << " ";
                    }
                }
                std::cout << "\n";
            
            };
            if (isFixedStartWeightsUsed)
            {
                std::cout << "Entered Start Weights:";
                printAllWeights(OldStartWeights);
                std::cout << "\n";
                std::cout << "Fixed Start Weights:";
                printAllWeights(StartWeights);

                std::cout << "You entered wrong weight data Fixed Start Weights going to be used.\n";
                std::cout << "Press Enter To Continue";
                char b;
                std::cin >> b;
            }



            for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)//Setting neurons for input layers
            {
                InputLayerNeurons[j] = new Neuron(ScreenNetsInputList[j], Input[0][j], LayerTypeEnum::InputLayer, j);
                 
            }

            for (unsigned int j = 0; j < ScreenNetsHiddenList.size(); j++)//Setting neurons for hidden layers
            {
                if (j == 0)
                    for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
                    {
                        HiddenLayerNeurons[j][i] = new Neuron(ScreenNetsHiddenList[j][i], InputLayerNeurons, StartWeights[j][i], LayerTypeEnum::HiddenLayer, i);
                         
                    }
                else
                    for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
                    {
                        HiddenLayerNeurons[j][i] = new Neuron(ScreenNetsHiddenList[j][i], HiddenLayerNeurons[j - 1], StartWeights[j][i], LayerTypeEnum::HiddenLayer, i, j);
                         
                    }
            } 

            for (unsigned int j = 0; j < ScreenNetsOutputList.size(); j++)//Setting neurons for output layers 
            {

                OutputLayerNeurons[j] = new Neuron(ScreenNetsOutputList[j], HiddenLayerNeurons[HiddenLayerNeurons.size() - 1], StartWeights[StartWeights.size() - 1][j], LayerTypeEnum::OutputLayer, j);
 
            }

        } 

        void PrintCost()
        {
    
            //CostPerEpochsAvg = CostPerEpochsAvg/
             
            StaticLogs = 
                "\033[97mCost Per Batch: " + std::to_string(Cost) + "\033[0m\n"
                "\033[92mCost Per Epochs: " + std::to_string(CostPerEpochs) + "\033[0m\n"
                "\033[93mCost Average: " + std::to_string(CostPerEpochsAverage) + "\033[0m\n"
                "\033[96mBatch Index: " + std::to_string(BatchIndex) +"\033[0m\n"
                "\033[91mTotal Epochs: " + std::to_string(EpochsIndex); "\033[0m";
        }

        void CalculateAverageCost()
        {  
            CostPerEpochs = std::accumulate(CostPerEpochsList.begin(), CostPerEpochsList.end(), 0.0) / CostPerEpochsList.size();


            CostPerEpochsAverageList.push_back(CostPerEpochs);
            CostPerEpochsAverage = std::accumulate(CostPerEpochsAverageList.begin(), CostPerEpochsAverageList.end(), 0.0) / CostPerEpochsAverageList.size();

            CostPerEpochsList.clear();
        }

        void IncreaseBatchIndex()
        {

            if (BatchIndex < Input.size()-1)
                BatchIndex++;
            else
            {
                BatchIndex = 0;
                EpochsIndex++;
                CalculateAverageCost();
            }
            for (unsigned int j = 0; j<InputLayerNeurons.size(); j++)
            {
                InputLayerNeurons[j]->SetOutputValueForInputNeuron(Settings::Input[BatchIndex][j]);
            } 
        }

        void ForwardPropagation()
        {
            //Disabling these 2 mfs fixing the error
            for (unsigned int j = 0; j < HiddenLayerNeurons.size(); j++)
                for (unsigned int i = 0; i < HiddenLayerNeurons[j].size(); i++)
                {
                    if (HiddenLayerNeurons[j][i] == nullptr)
                        throw std::invalid_argument("Output layer neuron is null");
                    HiddenLayerNeurons[j][i]->SetOutput();
                }

            for (unsigned int j = 0; j < OutputLayerNeurons.size(); j++)
            {
                Neuron* neuron = OutputLayerNeurons[j];
                if (neuron == nullptr)
                    throw std::invalid_argument("Output layer neuron is null");
                neuron->SetOutput();
            }
        }
        void BackPropagation()
        {
            auto CostFormula = [&](double y, double yExpected, unsigned int n) {

                double c = 0;
                for (unsigned int j = 0; j < n - 1; j++)
                {
                    Neuron* neuron = OutputLayerNeurons[j];
                    c += (neuron->ActivationOutput - ExpectedOutput[BatchIndex][j]) * (neuron->ActivationOutput - ExpectedOutput[BatchIndex][j]);
                }
                c += (yExpected - y) * (yExpected - y);
                c *= 1.0 / (double)n;
                return c;
            };

            auto derCostFormula = Math::derivative3(CostFormula);
            auto derActivationFormula = Math::derivative(Math::ActivationFormula);


            for (unsigned int j = 0; j < OutputLayerNeurons.size(); j++)//Output Neurons
            {
                Neuron* outputNeuron = OutputLayerNeurons[j];

                std::vector<double> newWeights1 = std::vector<double>(outputNeuron->ConnectedNeurons.size());
                std::vector<std::vector<std::vector<double>>> newWeights2 = ([&]() {//Layer->Neuron->Weights
                    std::vector<std::vector<std::vector<double>>> result = std::vector<std::vector<std::vector<double>>>(HiddenLayerSize.size());

                    for (unsigned int i = 0; i < HiddenLayerSize.size(); i++)
                    {
                        result[i] = std::vector<std::vector<double>>(HiddenLayerSize[i]);
                        for (unsigned int c = 0; c < HiddenLayerSize[i]; c++)
                            if (c == 0)
                                result[i][c] = std::vector<double>(InputLayerSize);
                            else
                                result[i][c] = std::vector<double>(HiddenLayerSize[c - 1]);

                    }

                    return result;
                })();


                Cost = CostFormula(outputNeuron->ActivationOutput, ExpectedOutput[BatchIndex][j], j + 1);
                double der1 = derCostFormula(outputNeuron->ActivationOutput, ExpectedOutput[BatchIndex][j], (double)j + 1);
                double der2 = derActivationFormula(outputNeuron->ActivationOutput);
                CostPerEpochsList.push_back(Cost);

                for (unsigned int i = 0; i < outputNeuron->ConnectedNeurons.size(); i++)//Last Hidden Layer - Output Layer
                {
                    Neuron* subNeuron = outputNeuron->ConnectedNeurons[i];
                    double derivatives1 = der1 * der2 * subNeuron->ActivationOutput;
                    double newWeight = outputNeuron->ConnectedNeurons[i]->OutputWeights[outputNeuron->NeuronIndex] + -derivatives1 * LearningRate;
                    if (isinf(newWeight))
                    {
                        std::cout << "\n" << newWeight << " " << outputNeuron->ConnectedNeurons[i]->OutputWeights[outputNeuron->NeuronIndex] << " " << -derivatives1 << " " << LearningRate;
                        throw std::invalid_argument("INFINITY FOUND!");
                    }
                    newWeights1[i] = newWeight;
                }

                for (unsigned int i = HiddenLayerNeurons.size() - 1; i != (unsigned int)(-1); i--)//Hidden layer - Hidden layer or Hidden layer - Input layer 
                    for (unsigned int c = 0; c < HiddenLayerNeurons[i].size(); c++)
                    {
                        Neuron* subNeuron = HiddenLayerNeurons[i][c]; 
                        for (unsigned int k = 0; k < subNeuron->ConnectedNeurons.size(); k++)
                        { 
                            //                        2 der                                                         //                         left handside output(Not activation) 
                            double derivatives2 = der1 * der2 * subNeuron->OutputWeights[k] * derActivationFormula(subNeuron->ActivationOutput) * subNeuron->ConnectedNeurons[k]->Output;
                            double subNewWeight = subNeuron->ConnectedNeurons[k]->OutputWeights[subNeuron->NeuronIndex] + -derivatives2 * LearningRate;
                            if (isinf(subNewWeight))
                            {
                                std::cout <<"\n" << subNewWeight << " " << subNeuron->ConnectedNeurons[k]->OutputWeights[subNeuron->NeuronIndex] <<" " << - derivatives2 <<" " << LearningRate;
                                throw std::invalid_argument("INFINITY FOUND!");
                            }
                            newWeights2[i][c][k] = subNewWeight;

                            //print("Expected: "+  s(ExpectedOutput[j])+ s(der1) + s(der2)+ s(der3) + s(derivatives2));
                        }
                    }
                 
                for (unsigned int i = 0; i < outputNeuron->ConnectedNeurons.size(); i++)//Last Hidden Layer - Output Layer 
                    outputNeuron->SetWeightAt(i, newWeights1[i]);
                for (unsigned int i = HiddenLayerNeurons.size() - 1; i != (unsigned int)(-1); i--)//Hidden layer - Hidden layer or Hidden layer - Input layer 
                    for (unsigned int c = 0; c < HiddenLayerNeurons[i].size(); c++)
                    {
                        Neuron* subNeuron = HiddenLayerNeurons[i][c];
                        for (unsigned int k = 0; k < subNeuron->ConnectedNeurons.size(); k++)
                            subNeuron->SetWeightAt(k, newWeights2[i][c][k]);
                    }
            } 
        }
    }
    using namespace NeuralNeurons;

    namespace InputThread
    {
        OutputTypeEnum Mode = OutputTypeEnum::null;
        std::list<OutputTypeEnum> PreviousModes = std::list<OutputTypeEnum>();
        unsigned int SelectedNeuronLayer = 0;
        unsigned int SelectedNeuronIndex = 0;
        bool IsEnteringLayerID = false;
        bool IsEnteringEpochsTarget = false;
        unsigned int SelectedTargetEpochs = 0;

        bool GoOneEmptyContinue = false;

        template<typename T>
        unsigned int GetLengthLinkedList(std::list<T>* list)
        {
            int len = 0;
            for (auto i = list->begin(); true; i++)
            {
                if (*i != 0)
                    len++;
                else
                    break;
            }

            return len;
        }
        template<typename T>
        void PrintLinkedList(std::list<T>* list)
        {
            for (auto i = list->begin(); true; i++)
            {
                if (*i != 0)
                {
                    std::cout << (int)*i << "\n";
                }
                else
                    break;
            }

        }

        void ModeSelected(OutputTypeEnum mode, bool skipPrev = false)
        {
            if (mode == OutputTypeEnum::ShowNetValue)
            {
                for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)
                    ScreenNetsInputList[j]->SetOutputType(OutputTypeEnum::ShowNetValue);

                for (unsigned int j = 0; j < ScreenNetsHiddenList.size(); j++)
                    for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
                        ScreenNetsHiddenList[j][i]->SetOutputType(OutputTypeEnum::ShowNetValue);

                for (unsigned int j = 0; j < ScreenNetsOutputList.size(); j++)
                    ScreenNetsOutputList[j]->SetOutputType(OutputTypeEnum::ShowNetValue);


                CurrentMode = "\033[92mSHOWING NEURON VALUES\033[0m";
                ModeSelection = "Press B to enter Target Epochs.";
            }
            else if (mode == OutputTypeEnum::ShowNetWeights)
            {
                if (ShowNetIDsInShowWeightMode)
                {
                    ModeSelected(OutputTypeEnum::ShowNetID, true);
                }
                else
                {
                    OutputTypeEnum selectedPreviousMode = OutputTypeEnum::null;
                    auto i = PreviousModes.end();
                    i--;
                    for (;; i--)
                    {
                        if (*i != 0)
                        {
                            if (*i != OutputTypeEnum::ShowNetWeights)
                            {
                                selectedPreviousMode = *i;
                                break;
                            }
                        }
                        else
                            break;
                    }
                    ModeSelected(selectedPreviousMode, true);
                }

                if (InputThread::SelectedNeuronLayer == 0)//For Input layers
                    for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)
                        ScreenNetsInputList[j]->SetOutputType(OutputTypeEnum::ShowNetWeights);
                else if (InputThread::SelectedNeuronLayer == TotalLayers - 1)//For Output layers
                {
                    for (unsigned int j = 0; j < NeuralNeurons::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->ConnectedNeurons.size(); j++)
                        NeuralNeurons::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->ConnectedNeurons[j]->ScreenNet->SetOutputType(OutputTypeEnum::ShowNetWeights);
                    NeuralNeurons::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->SetShowWeights();

                }
                else//Minus 1 for Hidden layers
                {
                    for (unsigned int j = 0; j < NeuralNeurons::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->ConnectedNeurons.size(); j++)
                        NeuralNeurons::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->ConnectedNeurons[j]->ScreenNet->SetOutputType(OutputTypeEnum::ShowNetWeights);
                    NeuralNeurons::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->SetShowWeights();

                }
                CurrentMode = "\033[93mSHOWING NEURON WEIGHTS\033[0m (For \033[92mLayer: " + std::to_string(SelectedNeuronLayer) + "\033[0m \033[95mIndex: " + std::to_string(SelectedNeuronIndex) + "\033[0m)";
                ModeSelection = "Press V to enter Layer ID";
            }
            else if (mode == OutputTypeEnum::ShowNetID)
            {
                for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)
                    ScreenNetsInputList[j]->SetOutputType(OutputTypeEnum::ShowNetID);

                for (unsigned int j = 0; j < ScreenNetsHiddenList.size(); j++)
                    for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
                        ScreenNetsHiddenList[j][i]->SetOutputType(OutputTypeEnum::ShowNetID);


                for (unsigned int j = 0; j < ScreenNetsOutputList.size(); j++)
                    ScreenNetsOutputList[j]->SetOutputType(OutputTypeEnum::ShowNetID);


                CurrentMode = "\033[97mSHOWING NEURON IDS\033[0m";
                ModeSelection = "";
            }


            if (!skipPrev)
            {
                if (PreviousModes.back() != Mode)
                {
                    PreviousModes.push_back(Mode);
                    if (GetLengthLinkedList(&PreviousModes) > 3)
                        PreviousModes.pop_front();

                }
                //PrintLinkedList(&PreviousModes);

            }
         
        }

        void InputMain()
        {
            if (InputThread::IsEnteringLayerID)
            {
                unsigned int numericalInput1 = 0;
                CurrentMode = "";
                StaticLogs = "";
                ModeSelection = "\033[92mEnter Neuron Layer in the range of [0, " + std::to_string(TotalLayers - 1) + "]: \033[0m";
                PrintLogs();
                while (true)
                {
                    if (!(std::cin >> numericalInput1))
                        std::cout << "Invalid input, please enter numbers...\n";
                    else if (numericalInput1 >= TotalLayers)
                        std::cout << "Out of range, please enter numbers in [0," + std::to_string(TotalLayers - 1) + "].\n";
                    else
                        break;
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                }

                InputThread::SelectedNeuronLayer = numericalInput1;


                unsigned int maxNeuronIndex = 0;

                if (InputThread::SelectedNeuronLayer == 0)
                    maxNeuronIndex = InputLayerSize;
                else if (InputThread::SelectedNeuronLayer == TotalLayers - 1)
                    maxNeuronIndex = OutputLayerSize;
                else
                    maxNeuronIndex = HiddenLayerSize[InputThread::SelectedNeuronLayer - 1];


                unsigned int numericalInput2 = 0;
                CurrentMode = "";
                StaticLogs = "";
                ModeSelection = "\033[95mEnter Neuron Index in the range of [0, " + std::to_string(maxNeuronIndex - 1) + "]: \033[0m";
                PrintLogs();

                while (true)
                {
                    if (!(std::cin >> numericalInput2))
                        std::cout << "Invalid input, please enter numbers...\n";
                    else if (numericalInput2 >= maxNeuronIndex)
                        std::cout << "Out of range, please enter numbers in [0," + std::to_string(maxNeuronIndex - 1) + "].\n";
                    else
                        break;
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                }

                InputThread::SelectedNeuronIndex = numericalInput2;
                InputThread::IsEnteringLayerID = false;
                GoOneEmptyContinue = true;

            }
            else if (InputThread::IsEnteringEpochsTarget)
            {
                unsigned int numericalInput1 = 0;
                CurrentMode = "";
                StaticLogs = "";
                ModeSelection = "\033[92mEnter Epochs Target in the range of ["+std::to_string(EpochsIndex)+", " + std::to_string(UINT_MAX) + "]: \033[0m";
                PrintLogs();
                while (true)
                {
                    if (!(std::cin >> numericalInput1))
                        std::cout << "Invalid input, please enter numbers...\n";
                    else if (numericalInput1 == UINT_MAX)
                        std::cout << "Out of range, please enter numbers in [" + std::to_string(EpochsIndex) + "," + std::to_string(TotalLayers - 1) + "].\n";
                    else
                        break;
                    std::cin.clear();
                    std::cin.ignore(10000, '\n');
                }

                InputThread::SelectedTargetEpochs = numericalInput1; 
                InputThread::IsEnteringEpochsTarget = false;
                GoOneEmptyContinue = true;
            }
        }
        void main()
        {
            while (1)
            {
                if (GetKeyState(toupper(KeyboardKeys::ShowNetValues)) & 0x8000)
                    Mode = (OutputTypeEnum::ShowNetValue);
                if (GetKeyState(toupper(KeyboardKeys::ShowNetWeights)) & 0x8000)
                    Mode = (OutputTypeEnum::ShowNetWeights);
                if (GetKeyState(toupper(KeyboardKeys::ShowNetID)) & 0x8000)
                    Mode = (OutputTypeEnum::ShowNetID); 


                if (!InputThread::IsEnteringLayerID && Mode == OutputTypeEnum::ShowNetWeights && (GetKeyState(toupper(KeyboardKeys::EnterLayer)) & 0x8000))
                    InputThread::IsEnteringLayerID = true;

                if (!InputThread::IsEnteringEpochsTarget && Mode == OutputTypeEnum::ShowNetValue && (GetKeyState(toupper(KeyboardKeys::EnterEpochsTarget)) & 0x8000))
                    InputThread::IsEnteringEpochsTarget = true;
            }
        }
        void OneEmptyContinued()
        {
            GoOneEmptyContinue = false;
        }
    }

    /// <summary>
    /// Second  
    /// </summary>
    void StartTeaching()
    {
        InputThread::Mode = (OutputTypeEnum::ShowNetValue);
        std::thread startThread(InputThread::main);
        char key;
        while(1)
        {
            InputThread::OneEmptyContinued();
            if(EpochsIndex < InputThread::SelectedTargetEpochs || key == KeyboardKeys::FeedForward || key == toupper(KeyboardKeys::FeedForward))
                ForwardPropagation();
            InputThread::ModeSelected(InputThread::Mode);
            InputThread::InputMain();


            

            if (EpochsIndex < InputThread::SelectedTargetEpochs || key == KeyboardKeys::FeedForward || key == toupper(KeyboardKeys::FeedForward))
            {
                BackPropagation();
                IncreaseBatchIndex();
            } 

            PrintCost();
            PrintScreen(); 
            if (1)
            {
                while (!(
                        InputThread::GoOneEmptyContinue
                        ||
                        EpochsIndex  <  InputThread::SelectedTargetEpochs 
                        ||
                         ( key = _getch()) &&
                         ( key == KeyboardKeys::FeedForward    || key == toupper(KeyboardKeys::FeedForward)
                        || key == KeyboardKeys::ShowNetValues  || key == toupper(KeyboardKeys::ShowNetValues)
                        || key == KeyboardKeys::ShowNetWeights || key == toupper(KeyboardKeys::ShowNetWeights)
                        || key == KeyboardKeys::ShowNetID      || key == toupper(KeyboardKeys::ShowNetID)
                        || key == KeyboardKeys::EnterEpochsTarget || key == toupper(KeyboardKeys::EnterEpochsTarget)
                        || key == KeyboardKeys::EnterLayer || key == toupper(KeyboardKeys::EnterLayer)
                        || key == KeyboardKeys::EmptyContinue
                )));

            }
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    /// <summary>
    /// Run this in main
    /// </summary>
    void Initialize()
    {

        SetScreen();
        InitializeNeurons();
        StartTeaching();


    }

    #pragma endregion 
}