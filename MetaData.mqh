//+------------------------------------------------------------------+
//|                                                     MetaData.mqh |
//|                                       Copyright 2022, Hedge Ltd. |
//|                                            https://www.hedge.com |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2022, hedge Ltd."
#property link          "https://www.hedge.com"
#property description   "10 seconds Meta Data"
#property version       "1.00"
//+--------------------------------------------------------------------------------------------------------------+
#include "structure.mqh"
#include "market_data.mqh"
//+--------------------------------------------------------------------------------------------------------------+

double getVariance(string symbol, bool TimeframeSeconds, ENUM_TIMEFRAMES period, int count)
{
   double mean = 0;
   double mean_squared = 0;
   
   //--- Calculation of the variance on a 2min range
   if(TimeframeSeconds)    //If we want to calculate the CC on MyMqlRates data (1 candlestick = 10sec).
   {
      MyMqlRates rates_symbol[];
      
      //--- Calcul of FromPast datetime (2min before)
      datetime FromPast;
      MqlDateTime daycurrent;
      TimeToStruct(TimeCurrent(), daycurrent);
      int hours = daycurrent.hour;
      int minutes = daycurrent.min;
      if(minutes<=3)
      {
         hours = daycurrent.hour - 1;
         minutes = 60 - 3 + daycurrent.min;
      }
      else minutes -= 3;
      FromPast = StringToTime((string)daycurrent.year + "." + (string)daycurrent.mon + "." + (string)daycurrent.day + " " + (string)hours + ":" + (string)minutes + ":" + (string)daycurrent.sec);
      
      //--- get MyMqlRates data
      int k1=0, i1=0;
      while (i1==0 && k1<3) //if data loading error: retry three times max
      {
         i1 = getRatesSeconde(symbol,FromPast,TimeCurrent(),rates_symbol,false,false,false,false,true,false,false,false,false,false,false,false,false);
         k1++;
      }
      
      //--- main loop
      for(int i=0; i<ArraySize(rates_symbol); i++)
      {
         mean+=rates_symbol[i].close;
         mean_squared += rates_symbol[i].close*rates_symbol[i].close;
      }
      mean /= ArraySize(rates_symbol);
      mean_squared /= ArraySize(rates_symbol);
   }
   //--- Calculation of the variance on a [M15,M30,H1,H4,D1] range
   else
   {
      //--- get MqlRates data
      MqlRates rates_symbol[];
      CopyRates(symbol,period,0,count,rates_symbol); //almost never error of loading on the highest timeframe
      ArraySetAsSeries(rates_symbol, true);
      
      //--- main loop
      for(int i=0; i<ArraySize(rates_symbol); i++)
      {
         mean+=rates_symbol[i].close;
         mean_squared += rates_symbol[i].close*rates_symbol[i].close;
      }
      mean /= ArraySize(rates_symbol);
      mean_squared /= ArraySize(rates_symbol);
   }
   
   //--- return variance value
   //printf("mean_squared = %f ; mean = %f", mean_squared, mean);
   return mean_squared - mean*mean;
}

double getCovariance(string symbol1, string symbol2, bool TimeframeSeconds, ENUM_TIMEFRAMES period, int count)
{
   double mean1 = 0, mean2 = 0, mean_product = 0;
   
   //--- Calculation of the covariance on a 2min range
   if(TimeframeSeconds)
   {
      //--- Calcul of FromPast datetime (2min before)
      datetime FromPast;
      MqlDateTime daycurrent;
      TimeToStruct(TimeCurrent(), daycurrent);
      int hours = daycurrent.hour;
      int minutes = daycurrent.min;
      if(minutes<=3)
      {
         hours = daycurrent.hour - 1;
         minutes = 60 - 3 + daycurrent.min;
      }
      else minutes -= 3;
      FromPast = StringToTime((string)daycurrent.year + "." + (string)daycurrent.mon + "." + (string)daycurrent.day + " " + (string)hours + ":" + (string)minutes + ":" + (string)daycurrent.sec);
      
      //--- get MyMqlRates data
      MyMqlRates rates_symbol1[], rates_symbol2[];
      int i1 = getRatesSeconde(symbol1,FromPast,TimeCurrent(),rates_symbol1,false,false,false,false,true,false,false,false,false,false,false,false,false);
      int i2 = getRatesSeconde(symbol2,FromPast,TimeCurrent(),rates_symbol2,false,false,false,false,true,false,false,false,false,false,false,false,false);
      if(i1 == 0 || i2 == 0) {printf("error i1 i2"); return 0;} //error in loading Bars or Ticks (cf. market_data file)
      
      
      //--- main loop
      for(int i=0; i<ArraySize(rates_symbol1); i++)
      {
         mean1+=rates_symbol1[i].close;
         mean2+=rates_symbol2[i].close;
         mean_product+= rates_symbol1[i].close*rates_symbol2[i].close;
      }
      mean1 /= ArraySize(rates_symbol1);
      mean2 /= ArraySize(rates_symbol2);
      mean_product /= ArraySize(rates_symbol1);
   }
   
   //--- Calculation of the covariance on a [M15,M30,H1,H4,D1] range
   else
   {
      //--- get MqlRates data
      MqlRates rates_symbol1[], rates_symbol2[];
      CopyRates(symbol1,period,0,count,rates_symbol1);
      ArraySetAsSeries(rates_symbol1, true);
      CopyRates(symbol2,period,0,count,rates_symbol2);
      ArraySetAsSeries(rates_symbol2, true);
   
      //--- main loop
      for(int i=0; i<MathMin(ArraySize(rates_symbol1),ArraySize(rates_symbol2)); i++)
      {
         mean1+=rates_symbol1[i].close;
         mean2+=rates_symbol2[i].close;
         mean_product+= rates_symbol1[i].close*rates_symbol2[i].close;
      }
      mean1 /= ArraySize(rates_symbol1);
      mean2 /= ArraySize(rates_symbol2);
      mean_product /= ArraySize(rates_symbol1);
   }
   
   //--- return covariance value
   return mean_product - mean1*mean2;
}

void getMostCorrelatedAssets(string symbol, string &MostCorrelatedAssets[], ENUM_TIMEFRAMES period, int count)
{
   int nb_symbols = SymbolsTotal(false);  //get number of all available symbols
   string symbol_name;
   
   //--- 15 most correlated symbols buffers
   double correlation_values[];
   string correlation_symbolName[];
   ArrayResize(correlation_values,15,0);
   ArrayResize(correlation_symbolName,15,0);
   double correlation;  //variable for saving the correlation value
   int min_index = 0;   //necessary to keep only the highest correlation (in absolute value)
   int j=0;             //index of the symbol to be calculated in the first loop
   
   //--- get MqlRates data
   MqlRates rates_symbol[]; //will be used to check if the market value of this symbol is not less than 0.1
                            //condition for a consistent result of the correlation calculation
   
   //--- Calcul of FromPast datetime (2min before)
   datetime FromPast;
   MqlDateTime daycurrent;
   TimeToStruct(TimeCurrent(), daycurrent);
   int hours = daycurrent.hour;
   int minutes = daycurrent.min;
   if(minutes<=3)
   {
      hours = daycurrent.hour - 1;
      minutes = 60 - 3 + daycurrent.min;
   }
   else minutes -= 3;
   FromPast = StringToTime((string)daycurrent.year + "." + (string)daycurrent.mon + "." + (string)daycurrent.day + " " + (string)hours + ":" + (string)minutes + ":" + (string)daycurrent.sec);
   
   //--- calculation of the number of bars normally obtained for this 2min period
   ulong nb_bars = ((ulong)TimeCurrent() - (ulong)FromPast)/60-1;
   
   //--- initialization of the 15 symbols
   for(int i=0; i<15 && j<nb_symbols ;i++)
   {  
      symbol_name = SymbolName(j, false);
      
      //--- first condition: the symbol must not start with a # (ETF and stocks) (because too many and unusable). The two symbols must not be backed by the same currency (obvious correlation)
      if(symbol_name[0] != '#' && symbol_name != symbol && SymbolInfoString(symbol,SYMBOL_CURRENCY_BASE) != SymbolInfoString(symbol_name,SYMBOL_CURRENCY_BASE) && SymbolInfoString(symbol,SYMBOL_CURRENCY_BASE) != SymbolInfoString(symbol_name,SYMBOL_CURRENCY_PROFIT))
      {
         int countBars_symbol = 0;
         int k = 0;
         while(countBars_symbol < (int)nb_bars && k<3) //if Bars loading error: retry three times max
         {
            countBars_symbol = Bars(symbol_name,PERIOD_M1,FromPast,TimeCurrent())-1;
            k++;
         }
         
         CopyRates(symbol_name,PERIOD_M1,0,1,rates_symbol);
         
         //--- second condition: the right number of bars must have been loaded and the market value must be greater than 0.1 (otherwise variance calculation error)
         if(countBars_symbol >= (int)nb_bars && rates_symbol[0].close>=0.1)
         {
            //---get the ith correlation value and the corresponding symbol
            correlation_symbolName[i] = symbol_name;
            correlation_values[i] = getCovariance(symbol,symbol_name, false, period, count)/(MathSqrt(getVariance(symbol, false, period, count)*getVariance(symbol_name, false, period, count)));
            
            //--- save the smallest correlation value.
            if(MathAbs(correlation_values[i])<MathAbs(correlation_values[min_index])) min_index = i;
            //printf("15 - Correlation with %s = %f", correlation_symbolName[i],correlation_values[i]);
            
         }
         else i--;
      }
      else i--;
      j++;
   }
   
   //ArrayPrint(correlation_values, 6, " ",0,WHOLE_ARRAY,NULL);
   //ArrayPrint(correlation_symbolName, _Digits, NULL,0,WHOLE_ARRAY,NULL);
   
   //--- calculation of next symbol correlations
   for(int i=j; i<nb_symbols; i++)
   {  
      symbol_name = SymbolName(i, false);
      
      //--- Same first condition
      if(symbol_name[0] != '#' && symbol_name != symbol && SymbolInfoString(symbol,SYMBOL_CURRENCY_BASE) != SymbolInfoString(symbol_name,SYMBOL_CURRENCY_BASE) && SymbolInfoString(symbol,SYMBOL_CURRENCY_BASE) != SymbolInfoString(symbol_name,SYMBOL_CURRENCY_PROFIT))
      {
         int countBars_symbol = 0;
         int c = 0;
         while(countBars_symbol < (int)nb_bars && c<3) //if Bars loading error: retry three times max
         {
            countBars_symbol = Bars(symbol_name,PERIOD_M1,FromPast,TimeCurrent())-1;
            c++;
         }
         CopyRates(symbol_name,PERIOD_M1,0,1,rates_symbol);
         
         //--- Same second condition
         if(countBars_symbol >= (int)nb_bars && rates_symbol[0].close>=0.1)
         {
            //--- get the ith correlation value and the corresponding symbol
            correlation = getCovariance(symbol,symbol_name, false, period, count)/(MathSqrt(getVariance(symbol, false, period, count)*getVariance(symbol_name, false, period, count)));
            
            //--- save the new correlation value if it is greater than the smallest saved value
            if(MathAbs(correlation)>MathAbs(correlation_values[min_index]))
            {
               correlation_values[min_index] = correlation;
               correlation_symbolName[min_index] = symbol_name;
               //printf("Correlation with %s = %f", symbol_name,(float)correlation);
               
               //--- search for the new smallest correlation value.
               for(int k=0; k<15; k++) if(MathAbs(correlation_values[k])<MathAbs(correlation_values[min_index])) min_index = k;
            }
         }
         //else printf("i = %d\t|%s|\t: bars could not be loaded - the market is probably closed", i, symbol_name);
      }
   }
   
   //ArrayPrint(correlation_values, 6, " ",0,WHOLE_ARRAY,NULL);
   //ArrayPrint(correlation_symbolName, _Digits, NULL,0,WHOLE_ARRAY,NULL);
   ArrayCopy(MostCorrelatedAssets,correlation_symbolName,0,0,WHOLE_ARRAY);
}

void getCorrelationCoefficient(string symbol, double &CC_value[][], string &CC_symbolsName[])
{
   //--- get the 15 symbols most correlated with the symbol
   string MostCorrelatedAssets[];
   getMostCorrelatedAssets(symbol,MostCorrelatedAssets,PERIOD_H1,100*24);
   ArrayCopy(CC_symbolsName,MostCorrelatedAssets,0,0,WHOLE_ARRAY);
   
   //--- multi timeframe correlation buffer
   double correlation_values[15][6];
   ArrayInitialize(correlation_values,0);
   
   //--- main loop
   for(int i=0; i<15; i++)
   {
      //--- 0:period=2*M1 ; 1:period=M15 ; 2:period=M30 ; 3:period=H1 ; 4:period=H4 ; 5:period=D1
      
      correlation_values[i][0] = getCovariance(symbol,MostCorrelatedAssets[i], true, PERIOD_CURRENT, 0)/(MathSqrt(getVariance(symbol, true, PERIOD_CURRENT, 0)*getVariance(MostCorrelatedAssets[i], true,PERIOD_CURRENT, 0)));
      while(correlation_values[i][0]>1.0 || correlation_values[i][0]<-1.0 || correlation_values[i][0]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][0] = getCovariance(symbol,MostCorrelatedAssets[i], true, PERIOD_CURRENT, 0)/(MathSqrt(getVariance(symbol, true, PERIOD_CURRENT, 0)*getVariance(MostCorrelatedAssets[i], true,PERIOD_CURRENT, 0)));
      if((string)correlation_values[i][0]=="-inf") correlation_values[i][0]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
      
      correlation_values[i][1] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 15)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 15)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 15)));
      while(correlation_values[i][1]>1.0 || correlation_values[i][1]<-1.0 || correlation_values[i][1]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][1] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 15)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 15)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 15)));
      if((string)correlation_values[i][1]=="-inf") correlation_values[i][1]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
      
      correlation_values[i][2] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 30)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 30)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 30)));
      while(correlation_values[i][2]>1.0 || correlation_values[i][2]<-1.0 || correlation_values[i][2]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][2] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 30)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 30)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 30)));
      if((string)correlation_values[i][2]=="-inf") correlation_values[i][2]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
      
      correlation_values[i][3] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 60)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 60)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 60)));
      while(correlation_values[i][3]>1.0 || correlation_values[i][3]<-1.0 || correlation_values[i][3]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][3] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M1, 60)/(MathSqrt(getVariance(symbol, false, PERIOD_M1, 60)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M1, 60)));
      if((string)correlation_values[i][3]=="-inf") correlation_values[i][3]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
      
      correlation_values[i][4] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M5, 48)/(MathSqrt(getVariance(symbol, false, PERIOD_M5, 48)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M5, 48)));
      while(correlation_values[i][4]>1.0 || correlation_values[i][4]<-1.0 || correlation_values[i][4]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][4] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M5, 48)/(MathSqrt(getVariance(symbol, false, PERIOD_M5, 48)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M5, 48)));
      if((string)correlation_values[i][4]=="-inf") correlation_values[i][4]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
      
      correlation_values[i][5] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M30, 48)/(MathSqrt(getVariance(symbol, false, PERIOD_M30, 48)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M30, 48)));
      while(correlation_values[i][5]>1.0 || correlation_values[i][5]<-1.0 || correlation_values[i][5]==0.0) //security if the correlation coefficient is inconsistent
         correlation_values[i][5] = getCovariance(symbol,MostCorrelatedAssets[i], false, PERIOD_M30, 48)/(MathSqrt(getVariance(symbol, false, PERIOD_M30, 48)*getVariance(MostCorrelatedAssets[i], false,PERIOD_M30, 48)));
      if((string)correlation_values[i][5]=="-inf") correlation_values[i][5]=1.0; //if the variance value is equal to 0 and thus the correlation is infinite, then replace by 1
   }
   ArrayCopy(CC_value,correlation_values,0,0,WHOLE_ARRAY);
}