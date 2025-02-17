// Neural Network.cpp : Defines the entry point for the application.
//

#include "Neural Network.h"
#include <vector>
#include "Dependencies/glm/glm.hpp"
#include "Dependencies/glm/gtx/string_cast.hpp"
#include <algorithm>
#include <chrono>
#include <thread> 
#include <Windows.h> //Windows.h for key inputs
#include <conio.h>
#include <limits>
#include <list>
#define s(x) std::to_string(x) + " "

 


//sigmoid function f(x) = 1/(1 + e^-x)
auto ActivationFormula = [&](double x) {
    return 1 / (1 + exp(-x));
};

#define LEARNING_RATE (double)0.1
const double delta = 0.0000001;
auto derivative = [](auto foo) {
    return [&](double x) {
        return (foo(x + delta) - foo(x)) / delta;
    };
};
auto derivative2 = [](auto foo) {
    return [&](double x, double y) {
        return (foo(x + delta, y) - foo(x,y)) / delta;
    };
};
auto derivative3 = [](auto foo) {
    return [&](double x, double y, double z) {
        return (foo(x + delta, y, z) - foo(x, y, z)) / delta;
    };
};  


const std::vector<std::vector<double>> Input = { {0.5, 0.3} };
const std::vector<unsigned int> HiddenLayerSize = { 2  };
const std::vector<std::vector<double>> ExpectedOutput = { {0.615} };
const unsigned int InputLayerSize = Input[0].size();
const unsigned int OutputLayerSize = ExpectedOutput[0].size();
const unsigned int TotalLayers = 2 + HiddenLayerSize.size();

//Size of std::vector<std::vector<std::vector<double>>> -> TotalLayers
//Size of std::vector<std::vector<double>> -> TotalNeurons
//Size of std::vector<double> -> TotalWeights 
const std::vector<std::vector<std::vector<double>>> StartWeights = std::vector<std::vector<std::vector<double>>>{

{ {0.7,0.4}, {0.3,0.6}/*1,1,1, 1,1,1,1,1, 1*/},//Input-Hidden
 

{ {0.55,0.45} }//Hidden-Output 

};

//const std::vector<std::vector<double>> StartWeights = std::vector<std::vector<double>>{
//
//{ 0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1,0.1, 0.1,0.1 },//Input-Hidden
//
//{ 0.1,0.1,0.1 },//Hidden1-Hidden2
//
//{ 0.1,0.1,0.1 },//Hidden2-Hidden3
//
//{ 0.1,0.1,0.1 }//Hidden-Output
//
//
//};

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
        
        ScreenNets(unsigned int BaseIndex, LayerTypeEnum LayerType, std::string NetID,unsigned int Index1, unsigned int Index2) : BaseIndex(BaseIndex), NetID(NetID), NetName("BaseIndex: " + std::to_string(BaseIndex))
        {
            if (NetID.length() != 4)
                throw std::invalid_argument("Net ID length isn't 4"); 
            ScreenNetsAllList.push_back(this);
            if(LayerType == LayerTypeEnum::InputLayer)
                ScreenNetsInputList[Index1] = (this);
            else if (LayerType == LayerTypeEnum::HiddenLayer)
            {
                ScreenNetsHiddenList[Index2][Index1] = (this);
            }
            else if (LayerType == LayerTypeEnum::OutputLayer)
                ScreenNetsOutputList[Index1] = (this);
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
    void PrintLogs()
    {
        if(CurrentMode.size() > 0)
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
        std::cout << Output   << "\033[F" << "\033[F";
        PrintLogs();

    };
    void SetScreen()
    {
        //Also Input Size

        unsigned int BlockRowSize = InputLayerSize;
        unsigned int RowSize = BlockRowSize*4;
        


        std::string fullOutput = "";
        unsigned int InputIndex = 0;
        std::vector<unsigned int> HiddenIndex = std::vector<unsigned int>(HiddenLayerSize.size());
        unsigned int OutputIndex = 0;

        unsigned int InputStartIndex = 0; 
        std::vector<unsigned int> HiddenStartIndex = std::vector<unsigned int>(HiddenLayerSize.size());
        for (unsigned int j = 0; j < HiddenStartIndex.size(); j++)
            HiddenStartIndex[j] = InputLayerSize - HiddenLayerSize[j];
        unsigned int OutputStartIndex = InputLayerSize - OutputLayerSize;


        std::vector<bool> HiddenPair = std::vector<bool>(HiddenLayerSize.size());
        for (unsigned int j = 0; j < HiddenPair.size(); j++)
            HiddenPair[j] = HiddenStartIndex[j] % 2;

        bool OutputPair = InputLayerSize % 2 ? !(OutputLayerSize % 2) : (OutputLayerSize % 2);


        unsigned int TBaseIndex = 0; 
        for (unsigned int i = 0; i < BlockRowSize* 2; i++)
        {
            std::string oneRow = "";  

            bool DrawInputNet = false, DrawOutputNet = false;
            std::vector<bool> DrawHiddenNet = std::vector<bool>(HiddenLayerSize.size());
            { 
                if (i % 2 == 0)
                {
                    DrawInputNet = true;
                    InputIndex++;
                }
                else
                    DrawInputNet = false;
            }
            {
                for(unsigned int c = 0; c < HiddenStartIndex.size(); c++)
                    if (i >= HiddenStartIndex[c] && HiddenIndex[c] < HiddenLayerSize[c])
                    {

                        if (i % 2 == HiddenPair[c])
                        {
                            DrawHiddenNet[c] = true;
                            HiddenIndex[c]++;
                        }
                         
                    }
            }

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

            {//Drawing Net
                        
                unsigned int netSizeForRow = 1;
                unsigned int IndexForNet = 0;
                oneRow = "      ";
                for (unsigned int k = 0; k < 2 + HiddenLayerSize.size(); k++)
                    if ((DrawInputNet && k == 0) ||    
                        (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1 ) ||
                        (k != 0 && DrawHiddenNet[k-1]))
                        oneRow += "\033[94m+------+  \033[0m    ";
                    else 
                        oneRow += "              ";
                    
                oneRow += "\n";
                    
                oneRow += "      ";
                for (unsigned int k = 0; k < 2 + HiddenLayerSize.size(); k++)
                {    
                     
                    LayerTypeEnum LayerType = (DrawInputNet && k == 0) ? LayerTypeEnum::InputLayer : (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1) ? LayerTypeEnum::OutputLayer : (k != 0 && DrawHiddenNet[k - 1]) ? LayerTypeEnum::HiddenLayer : LayerTypeEnum::NonLayer;
                    unsigned int Value = (DrawInputNet && k == 0) ? InputIndex : (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1) ? OutputIndex : (k != 0 && DrawHiddenNet[k - 1]) ? HiddenIndex[k-1] : (unsigned int)-1;
                    bool kIs2Digits = true;
                    if (k < 10)
                        kIs2Digits = false;

                    if (Value != (unsigned int)-1)
                    { 
                        if (Value - 1 < 10)
                        {
                            if(kIs2Digits)
                                oneRow += "|\033[35m" + (std::string)std::to_string(k) + "_" + (std::string)std::to_string(Value - 1) + "\033[0m |      ";//Total Length: 15
                            else
                                oneRow += "| " + (std::string)std::to_string(k) + "__" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                        }
                        else
                        {
                            if(kIs2Digits)
                                oneRow += "| " + (std::string)std::to_string(k) + "" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                            else 
                                oneRow += "| " + (std::string)std::to_string(k) + "_" + (std::string)std::to_string(Value - 1) + " |      ";//Total Length: 15
                        }
                        if (Output.size() + oneRow.size() - 13 < 0)
                            throw std::invalid_argument("error");
                        TBaseIndex = (Output.size() + oneRow.size() - 13);

                        new ScreenNets(TBaseIndex, LayerType, oneRow.substr(oneRow.length() - 12, 4),Value - 1, k-1); 
                    }
                    else
                        oneRow += "              ";
                }
                oneRow += "\n"; 

                oneRow += "      ";
                for (unsigned int k = 0; k < 2 + HiddenLayerSize.size(); k++)
                    if ((DrawInputNet && k == 0) ||
                        (DrawOutputNet && k == 2 + HiddenLayerSize.size() - 1) ||
                        (k != 0 && DrawHiddenNet[k - 1]))
                        oneRow += "\033[35m+------+  \033[0m    ";
                    else
                        oneRow += "              ";

                oneRow += "\n";
            }
            Output += oneRow; 
        }         

        IsScreenSet = true;
    }

}
using namespace Screen;
namespace NeuralNetwork
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
        bool IsInputORBIASNeuron;

        /// <summary>
        /// Setting hidden layer neurons or output neuron.
        /// </summary>
        /// <param name="ConnectedNeurons"></param>
        /// <param name="Weights"></param>
        /// <param name="NeuronLength"></param>
        Neuron(Screen::ScreenNets* ScreenNet, std::vector<Neuron*> ConnectedNeurons, std::vector<double> Weights,LayerTypeEnum LayerType, unsigned short NeuronIndex, unsigned short NeuronLayerIndex = 0)
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
            this->IsInputORBIASNeuron = false;
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
        Neuron(Screen::ScreenNets* ScreenNet, double Output, LayerTypeEnum LayerType, unsigned short NeuronIndex) : ScreenNet(ScreenNet), Output(Output), LayerType(LayerType), IsInputORBIASNeuron(true), IsOutputSet(true), NeuronIndex(NeuronIndex), NeuronLayerIndex(0), IsValid(true) {

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
                if (neuron->IsInputORBIASNeuron)
                {
                    Output += neuron->Output * neuron->OutputWeights[this->NeuronIndex];
                     
                }
                else
                {
                    Output += neuron->ActivationOutput * neuron->OutputWeights[this->NeuronIndex];
                }
            }
        }
        void ActivationFunction()
        {
            ActivationOutput = ActivationFormula(Output);
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
            for(unsigned int j = 0; j < this->ConnectedNeurons.size(); j++)
                this->ConnectedNeurons[j]->ScreenNet->SelectWeight(this->NeuronIndex);
        }
        void SetOutputValueForInputNeuron(double Value)
        {
            if (!IsInputORBIASNeuron)
                throw std::invalid_argument("This isn't Input neuron");

            Output = Value;
            SetOutput();
        }
        void SetOutput()
        { 
            if (IsInputORBIASNeuron)
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

    unsigned int DataIndex = 0;
    std::vector<double> CostDifference = std::vector<double>(ExpectedOutput.size()); 

    void InitializeNeurons()
    {

        if (InputLayerSize != ScreenNetsInputList.size())
            throw std::invalid_argument("Sizes don't match.");


        if (StartWeights.size() != TotalLayers - 1)
        {
            std::cout << StartWeights.size() << " " << TotalLayers;
            throw std::invalid_argument("Sizes don't match.");
        }


        if (StartWeights[0].size() != InputLayerSize)
        {
            std::cout << StartWeights[0].size() << " " << InputLayerSize;
            throw std::invalid_argument("Sizes don't match.");
        }

        for (uint32_t j = 1; j < StartWeights.size()-1; j++)
        {
            if (StartWeights[j].size() != HiddenLayerSize[j - 1])
            {
                std::cout << StartWeights[j].size() << " " << HiddenLayerSize[j - 1];
                throw std::invalid_argument("Sizes don't match.");
            } 
        }

        if (StartWeights[StartWeights.size()-1].size() != OutputLayerSize)
        {
            std::cout << StartWeights[StartWeights.size() - 1].size() << " " << OutputLayerSize;
            throw std::invalid_argument("Sizes don't match.");
        }

        for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)//Setting neurons for input layers
        { 
            Neuron* someNeuron = new Neuron(ScreenNetsInputList[j], Input[0][j], LayerTypeEnum::InputLayer, j);

            InputLayerNeurons[j] = someNeuron; 
        }

         
        for (unsigned int j = 0; j < ScreenNetsHiddenList.size(); j++)//Setting neurons for hidüden layers
        {
            if (j == 0)
                for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
                {
                    Neuron* someNeuron = new Neuron(ScreenNetsHiddenList[j][i], InputLayerNeurons, StartWeights[j][i], LayerTypeEnum::HiddenLayer, i);

                    HiddenLayerNeurons[j][i] = someNeuron;
                }
           else
               for (unsigned int i = 0; i < ScreenNetsHiddenList[j].size(); i++)
               {
                   Neuron* someNeuron = new Neuron(ScreenNetsHiddenList[j][i], HiddenLayerNeurons[j - 1], StartWeights[j][i], LayerTypeEnum::HiddenLayer, i, j);
           
                   HiddenLayerNeurons[j][i] = someNeuron;
               }
        }
         
         
        

        for (unsigned int j = 0; j < ScreenNetsOutputList.size(); j++)//Setting neurons for output layers 
        {
            Neuron* someNeuron = new Neuron(ScreenNetsOutputList[j], HiddenLayerNeurons[HiddenLayerNeurons.size() - 1], StartWeights[StartWeights.size()-1][j], LayerTypeEnum::OutputLayer, j);

            OutputLayerNeurons[j] = someNeuron;
        }

    }
    std::vector<double> CostValueList;

    void PrintCost()
    {
        StaticLogs = ("Loss: " + s(Cost) + " Data Index: " + s(DataIndex));
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
            for (unsigned int j = 0; j < n-1; j++)
            {
                Neuron* neuron = OutputLayerNeurons[j];
                c += (neuron->ActivationOutput - ExpectedOutput[DataIndex][j]) * (neuron->ActivationOutput - ExpectedOutput[DataIndex][j]);
            }
            c += (yExpected - y) * (yExpected - y);
            c *= 1.0 / (double)n;
            return c;
        };

        auto derCostFormula = derivative3(CostFormula);
        auto derActivationFormula = derivative(ActivationFormula);
         

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
                            result[i][c] = std::vector<double>(HiddenLayerSize[c-1]);

                }

                return result;
                })();


            Cost = CostFormula(outputNeuron->ActivationOutput, ExpectedOutput[DataIndex][j], j + 1);
            double der1 = derCostFormula(outputNeuron->ActivationOutput, ExpectedOutput[DataIndex][j], (double)j + 1);
            double der2 = derActivationFormula(outputNeuron->ActivationOutput);

            for (unsigned int i = 0; i < outputNeuron->ConnectedNeurons.size(); i++)//Last Hidden Layer - Output Layer
            {
                Neuron* subNeuron = outputNeuron->ConnectedNeurons[i]; 
                double derivatives1 = der1 * der2 * subNeuron->ActivationOutput;
                double newWeight = outputNeuron->ConnectedNeurons[i]->OutputWeights[outputNeuron->NeuronIndex] + -derivatives1 * LEARNING_RATE; 
                newWeights1[i] = newWeight;
            } 

            for (unsigned int i = HiddenLayerNeurons.size()-1; i != (unsigned int)(-1); i--)//Hidden layer - Hidden layer or Hidden layer - Input layer 
                for (unsigned int c = 0; c < HiddenLayerNeurons[i].size(); c++)
                {
                    Neuron* subNeuron = HiddenLayerNeurons[i][c];
                    for (unsigned int k = 0; k < subNeuron->ConnectedNeurons.size(); k++)
                    {
                        //                        2 der                                                         //                         left handside output(Not activation) 
                        double derivatives2 = der1 * der2 * subNeuron->OutputWeights[k] * derActivationFormula(subNeuron->ActivationOutput) * subNeuron->ConnectedNeurons[k]->Output;
                        double subNewWeight = subNeuron->ConnectedNeurons[k]->OutputWeights[subNeuron->NeuronIndex] + -derivatives2 * LEARNING_RATE; 
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
            PrintCost();
        }

    }
}

/*

        for (unsigned int j = 0; j < OutputLayerNeurons.size(); j++)//Output Neurons
        {
            Neuron* neuron = OutputLayerNeurons[j];

            Cost = CostFormula(neuron->ActivationOutput, ExpectedOutput[DataIndex][j], j + 1);
            double der1 = derCostFormula(neuron->ActivationOutput, ExpectedOutput[DataIndex][j], j + 1);
            double der2 = derActivationFormula(neuron->ActivationOutput);
            for (unsigned int i = 0; i < neuron->ConnectedNeurons.size(); i++)//Hidden Neurons-Output Neurons
            {
                Neuron* subNeuron = neuron->ConnectedNeurons[i];
                double derivatives1 = der1 * der2 * subNeuron->ActivationOutput;
                double newWeight = neuron->ConnectedNeurons[i]->OutputWeights[neuron->NeuronIndex] + -derivatives1 * LEARNING_RATE;
                neuron->SetWeightAt(i, newWeight);
                for (unsigned int k = 0; k < subNeuron->ConnectedNeurons.size(); k++)//Input Neurons
                {
                    double derivatives2 = der1 * der2 * derActivationFormula(subNeuron->ActivationOutput) * newWeight * subNeuron->ConnectedNeurons[k]->Output;
                    double subNewWeight = subNeuron->ConnectedNeurons[k]->OutputWeights[subNeuron->NeuronIndex] + -derivatives2 * LEARNING_RATE;
                    subNeuron->SetWeightAt(k, subNewWeight);
                    //print("Expected: "+  s(ExpectedOutput[j])+ s(der1) + s(der2)+ s(der3) + s(derivatives2));
                }
            }
            PrintCost();
        }

*/
namespace InputThread
{ 
    OutputTypeEnum Mode = OutputTypeEnum::null;
    std::list<OutputTypeEnum> PreviousModes = std::list<OutputTypeEnum>();
    unsigned int SelectedNeuronLayer = 0;
    unsigned int SelectedNeuronIndex = 0;
    bool IsEnteringLayerID = false; 
     

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


            CurrentMode = "Showing Neuron Values";
            ModeSelection = "";
        }
        else if (mode == OutputTypeEnum::ShowNetWeights)
        {
            OutputTypeEnum selectedPreviousMode = OutputTypeEnum::null;
            auto i = PreviousModes.end();
            i--;
            for (;;i--)
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

            if (InputThread::SelectedNeuronLayer == 0)//For Input layers
                for (unsigned int j = 0; j < ScreenNetsInputList.size(); j++)
                    ScreenNetsInputList[j]->SetOutputType(OutputTypeEnum::ShowNetWeights);
            else if (InputThread::SelectedNeuronLayer == TotalLayers - 1)//For Output layers
            {
                for (unsigned int j = 0; j < NeuralNetwork::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->ConnectedNeurons.size(); j++)
                    NeuralNetwork::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->ConnectedNeurons[j]->ScreenNet->SetOutputType(OutputTypeEnum::ShowNetWeights);
                NeuralNetwork::OutputLayerNeurons[InputThread::SelectedNeuronIndex]->SetShowWeights();

            }
            else//Minus 1 for Hidden layers
            {
                for (unsigned int j = 0; j < NeuralNetwork::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->ConnectedNeurons.size(); j++)
                    NeuralNetwork::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->ConnectedNeurons[j]->ScreenNet->SetOutputType(OutputTypeEnum::ShowNetWeights);
                NeuralNetwork::HiddenLayerNeurons[InputThread::SelectedNeuronLayer - 1][InputThread::SelectedNeuronIndex]->SetShowWeights();

            }
            CurrentMode = "Showing Neuron Weights For \033[92mLayer: " + std::to_string(SelectedNeuronLayer) + "\033[0m \033[95mIndex: " + std::to_string(SelectedNeuronIndex)+"\033[0m";
            ModeSelection = "Press V to enter layer ID";
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


            CurrentMode = "Showing Neuron IDs";
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
            ModeSelection =  "\033[92mEnter Neuron Layer in the range of [0, "+std::to_string(TotalLayers)+"): \033[0m";
            PrintLogs();
            while (true)
            {
                if (!(std::cin >> numericalInput1)) 
                    std::cout << "Invalid input, please enter numbers...\n";
                else if(numericalInput1 >= TotalLayers) 
                    std::cout << "Out of range, please enter numbers in [0," + std::to_string(TotalLayers) + ").\n";
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
                maxNeuronIndex = HiddenLayerSize[InputThread::SelectedNeuronLayer-1];
            

            unsigned int numericalInput2 = 0; 
            CurrentMode = "";
            StaticLogs = "";
            ModeSelection = "\033[95mEnter Neuron Index in the range of [0, "+std::to_string(maxNeuronIndex)+"): \033[0m";
            PrintLogs();

            while (true)
            {
                if (!(std::cin >> numericalInput2))
                    std::cout << "Invalid input, please enter numbers...\n";
                else if (numericalInput2 >= maxNeuronIndex) 
                    std::cout << "Out of range, please enter numbers in [0,"+std::to_string(maxNeuronIndex)+").\n";
                else
                    break;
                std::cin.clear();
                std::cin.ignore(10000, '\n');
            }

            InputThread::SelectedNeuronIndex = numericalInput2;
            InputThread::IsEnteringLayerID = false;

        } 
    }
    void main()
    {
        while (1)
        { 
            if (GetKeyState('A') & 0x8000) 
                Mode = (OutputTypeEnum::ShowNetValue);
            if(GetKeyState('S') & 0x8000) 
                Mode = (OutputTypeEnum::ShowNetWeights);
            if (GetKeyState('D') & 0x8000) 
                Mode = (OutputTypeEnum::ShowNetID);
            if (!InputThread::IsEnteringLayerID && (GetKeyState('V') & 0x8000 )) 
                IsEnteringLayerID = true;  
        }
    }
}

using namespace NeuralNetwork; 


int main()
{   
    SetScreen(); 


    InitializeNeurons();
     

    InputThread::Mode = (OutputTypeEnum::ShowNetValue);
    std::thread startThread(InputThread::main);  
    for(unsigned int p = 0; ; p++)
    {  
         
        ForwardPropagation(); 
        InputThread::ModeSelected(InputThread::Mode);
        InputThread::InputMain();


        PrintScreen();  
 
        if (1)
            system("pause");
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

        BackPropagation();
    }
    return 0; 
}


