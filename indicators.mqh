//+------------------------------------------------------------------+
//|                                                   indicators.mqh |
//|                                       Copyright 2022, Hedge Ltd. |
//|                                            https://www.hedge.com |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2022, hedge Ltd."
#property link          "https://www.hedge.com"
#property description   "10 seconds Indicator data"
#property version       "1.00"
//+--------------------------------------------------------------------------------------------------------------+
#include "structure.mqh"
#include <MovingAverages.mqh>
//+--------------------------------------------------------------------------------------------------------------+

void getADX(int ADX_Period, double &ADX[], double &PDI[], double &NDI[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(ADX) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     ArraySize(PDI) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     ArraySize(NDI) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     ADX, PDI, NDI will be of the TimeSeries form : ADX[0] is the most recent value.
     
     int InpPeriodADX=14; // Period ADX
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total<=200)
     {
      printf("Erreur(indicators.mqh:getADX) : size of RatesSecond > 200.");
      ArrayInitialize(ADX, 0.0);
      ArrayInitialize(PDI, 0.0);
      ArrayInitialize(NDI, 0.0);
      return;
     }
   
   int Rates_marge = 150;    // Number of candles needed to calculate correct ADX values
   
   //--- indicator buffers
   double DIP_Buffer[];
   ArrayResize(DIP_Buffer, rates_total, 0);
   double DIM_Buffer[];
   ArrayResize(DIM_Buffer, rates_total, 0);
   double ADX_Buffer[];
   ArrayResize(ADX_Buffer, rates_total, 0);
   
   double Tmp_Buffer[];
   ArrayResize(Tmp_Buffer, rates_total, 0);
   double DP_Buffer[];
   ArrayResize(DP_Buffer, rates_total, 0);
   double DM_Buffer[];
   ArrayResize(DM_Buffer, rates_total, 0);

   //--- main cycle
   for(int i=1; i<rates_total; i++)
   {
      //--- get some data
      double high_price = RatesSeconde[rates_total-i-1].high;
      double prev_high = RatesSeconde[rates_total-i].high;
      double low_price = RatesSeconde[rates_total-i-1].low;
      double prev_low  = RatesSeconde[rates_total-i].low;
      double prev_close = RatesSeconde[rates_total-i].close;
      
      //--- fill main positive and main negative buffers
      double MP = high_price-prev_high;
      double MM = prev_low-low_price;
      
      double DMP = (MP>MM && MP>0) ? MP : 0.0;
      double DMM = (MM>MP && MM>0) ? MM : 0.0;
      
      //--- define TR
      double TR = MathMax(MathMax(high_price-low_price, MathAbs(high_price-prev_close)), MathAbs(low_price-prev_close));
      if(TR != 0.0)
        {
         DP_Buffer[i] = 100.0 * DMP/TR;
         DM_Buffer[i] = 100.0 * DMM/TR;
        }
      else
        {
         DP_Buffer[i] = 0.0;
         DM_Buffer[i] = 0.0;
        }
      //--- fill smoothed positive and negative buffers
      DIP_Buffer[i] = ExponentialMA(i,ADX_Period,DIP_Buffer[i-1],DP_Buffer);
      DIM_Buffer[i] = ExponentialMA(i,ADX_Period,DIM_Buffer[i-1],DM_Buffer);
      //--- fill ADXTmp buffer
      double tmp = DIP_Buffer[i]+DIM_Buffer[i];
      if(tmp != 0.0)
         tmp = 100.0*MathAbs((DIP_Buffer[i]-DIM_Buffer[i])/tmp);
      else
         tmp = 0.0;
      Tmp_Buffer[i] = tmp;
      //--- fill smoothed ADX buffer
      ADX_Buffer[i] = ExponentialMA(i,ADX_Period,ADX_Buffer[i-1],Tmp_Buffer);
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(ADX, ADX_Buffer, 0, 200, ArraySize(ADX_Buffer) - 200 - 1);
   ArrayReverse(ADX, 0, ArraySize(ADX));
   ArrayCopy(PDI, DIP_Buffer, 0, 200, ArraySize(DIP_Buffer) - 200 - 1);
   ArrayReverse(PDI, 0, ArraySize(PDI));
   ArrayCopy(NDI, DIM_Buffer, 0, 200, ArraySize(DIM_Buffer) - 200 - 1);
   ArrayReverse(NDI, 0, ArraySize(NDI));
}

//+--------------------------------------------------------------------------------------------------------------+

void getRSI(int RSI_Period, double &RSI[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(RSI) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     RSI will be of the TimeSeries form : RSI[0] is the most recent value.
     
     int RSI_Period=14; // Period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total<=200)
   {
      printf("Erreur(indicators.mqh:getRSI) : size of RatesSecond > 200.");
      ArrayInitialize(RSI, 0.0);
      return;
   }
   
   //--- indicator buffers
   double    ExtRSIBuffer[];
   ArrayResize(ExtRSIBuffer,rates_total, 0);
   double    ExtPosBuffer[];
   ArrayResize(ExtPosBuffer,rates_total, 0);
   double    ExtNegBuffer[];
   ArrayResize(ExtNegBuffer,rates_total, 0);
   ExtRSIBuffer[0]=0.0;    //first RSIPeriod values of the indicator are not calculated
   ExtPosBuffer[0]=0.0;
   ExtNegBuffer[0]=0.0;
   double    sum_pos=0.0;
   double    sum_neg=0.0;
   
   //--- price buffers
   double    price[];
   ArrayResize(price, rates_total, 0);
   for(int i=0; i<rates_total; i++) price[rates_total-1-i] = RatesSeconde[i].close;
      
   //--- preliminary calculations
   for(int i=1; i<= RSI_Period; i++){
         ExtRSIBuffer[i] = 0.0;
         ExtPosBuffer[i] = 0.0;
         ExtNegBuffer[i] = 0.0;
         double diff = price[i] - price[i-1];
         sum_pos += (diff>0?diff:0);
         sum_neg += (diff<0?-diff:0);
        }
   
   //--- calculate first visible value
   ExtPosBuffer[RSI_Period]=sum_pos/RSI_Period;
   ExtNegBuffer[RSI_Period]=sum_neg/RSI_Period;
   if(ExtNegBuffer[RSI_Period]!=0.0)
      ExtRSIBuffer[RSI_Period]=100.0-(100.0/(1.0+ExtPosBuffer[RSI_Period]/ExtNegBuffer[RSI_Period]));
   else
     {
      if(ExtPosBuffer[RSI_Period]!=0.0)
         ExtRSIBuffer[RSI_Period]=100.0;
      else
         ExtRSIBuffer[RSI_Period]=50.0;
     }
   
   //--- the main loop of calculations
   for(int i=15; i<rates_total; i++)
   {
      double diff=price[i]-price[i-1];
      ExtPosBuffer[i]=(ExtPosBuffer[i-1]*(RSI_Period-1)+(diff>0.0?diff:0.0))/RSI_Period;
      ExtNegBuffer[i]=(ExtNegBuffer[i-1]*(RSI_Period-1)+(diff<0.0?-diff:0.0))/RSI_Period;
      if(ExtNegBuffer[i]!=0.0)
         ExtRSIBuffer[i]=100.0-100.0/(1+ExtPosBuffer[i]/ExtNegBuffer[i]);
      else
        {
         if(ExtPosBuffer[i]!=0.0)
            ExtRSIBuffer[i]=100.0;
         else
            ExtRSIBuffer[i]=50.0;
        }
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(RSI, ExtRSIBuffer, 0, 200, ArraySize(ExtRSIBuffer) - 200 - 1);
   ArrayReverse(RSI, 0, ArraySize(RSI));
}

//+--------------------------------------------------------------------------------------------------------------+

void getMomentum(int MOM_Period, double &MOM[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(MOM) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     MOM will be of the TimeSeries form : MOM[0] is the most recent value.
     
     int InpMomentumPeriod=14; // Period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200)
   {
      printf("Erreur(indicators.mqh:getMomentum) : size of RatesSecond > 200.");
      ArrayInitialize(MOM, 0.0);
      return;
   }
   
   //--- indicator buffers
   double    MOM_Buffer[];
   ArrayResize(MOM_Buffer,rates_total, 0);
   
   //--- price buffer
   double    price[];
   ArrayResize(price, rates_total, 0);
   for(int i=0; i<rates_total; i++) price[rates_total-1-i] = RatesSeconde[i].close;

   //--- main cycle
   for(int i=MOM_Period; i<rates_total; i++) MOM_Buffer[i]=price[i]*100/price[i-MOM_Period];
   
   //--- copy correct data into the buffer
   ArrayCopy(MOM, MOM_Buffer, 0, 200, ArraySize(MOM_Buffer) - 200 - 1);
   ArrayReverse(MOM, 0, ArraySize(MOM));
}

//+--------------------------------------------------------------------------------------------------------------+

void getCCI(int CCI_Period, double &CCI[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(CCI) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     CCI will be of the TimeSeries form : CCI[0] is the most recent value.
     
     int  InpCCIPeriod=14; // Period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200)
   {
      printf("Erreur(indicators.mqh:getCCI) : size of RatesSecond > 200.");
      ArrayInitialize(CCI, 0.0);
      return;
   }
   
   //--- indicator buffers
   double     ExtSPBuffer[];
   ArrayResize(ExtSPBuffer,rates_total, 0);
   double     ExtDBuffer[];
   ArrayResize(ExtDBuffer,rates_total, 0);
   double     ExtMBuffer[];
   ArrayResize(ExtMBuffer,rates_total, 0);
   double     ExtCCIBuffer[];
   ArrayResize(ExtCCIBuffer,rates_total, 0);
   
   double     ExtMultiplyer = 0.015/CCI_Period;
   
   //--- price buffer
   double    price[];
   ArrayResize(price, rates_total, 0);
   for(int i=0; i<rates_total; i++) price[rates_total-1-i] = RatesSeconde[i].close;
   
   //--- main cycle
   for(int i=CCI_Period; i<rates_total; i++)
     {
      ExtSPBuffer[i]=SimpleMA(i,CCI_Period,price);
      //--- calculate D
      double tmp_d=0.0;
      for(int j=0; j<CCI_Period; j++)
         tmp_d+=MathAbs(price[i-j]-ExtSPBuffer[i]);
      ExtDBuffer[i]=tmp_d*ExtMultiplyer;
      //--- calculate M
      ExtMBuffer[i]=price[i]-ExtSPBuffer[i];
      //--- calculate CCI
      if(ExtDBuffer[i]!=0.0)
         ExtCCIBuffer[i]=ExtMBuffer[i]/ExtDBuffer[i];
      else
         ExtCCIBuffer[i]=0.0;
     }
     
   //--- copy correct data into the buffer
   ArrayCopy(CCI, ExtCCIBuffer, 0, 200, ArraySize(ExtCCIBuffer) - 200 - 1);
   ArrayReverse(CCI, 0, ArraySize(CCI));
}

//+--------------------------------------------------------------------------------------------------------------+

void getAO(int FAST_PERIOD, int SLOW_PERIOD, double &AO[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(AO) = ArraySize(RatesSedonde) - 200 - 1.     //"-200":precision margin ; "-1":starts at the second last
     
     AO will be of the TimeSeries form : AO[0] is the most recent value.
     
     //--- default parameters
     FAST_PERIOD = 5;         // Fast SMA period
     SLOW_PERIOD = 34;        // Slow SMA period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200)
   {
      printf("Erreur(indicators.mqh:getAO) : size of RatesSecond > 200.");
      ArrayInitialize(AO, 0.0);
      return;
   }
   
   //--- price buffer
   double    median[];
   ArrayResize(median, rates_total, 0);
   for(int i=0; i<rates_total; i++) median[rates_total-1-i] = (RatesSeconde[i].high + RatesSeconde[i].low)/2;
   
   //--- indicator buffers
   double     ExtAOBuffer[];
   ArrayResize(ExtAOBuffer,rates_total, 0);
   double     ExtFastBuffer[];
   ArrayResize(ExtFastBuffer,rates_total, 0);
   double     ExtSlowBuffer[];
   ArrayResize(ExtSlowBuffer,rates_total, 0);
   double TempVarBuffer[]; //temporary buffer for the use of the getSMA function
   
   //--- buffer for MAs
   getSMA(FAST_PERIOD,0,ExtFastBuffer,TempVarBuffer,median);
   getSMA(SLOW_PERIOD,0,ExtSlowBuffer,TempVarBuffer,median);
     
   //--- first calculation
   int i;
   for(i=0; i<33; i++) ExtAOBuffer[i]=0.0;

   //--- main loop of calculations
   for(i=33; i<rates_total; i++) ExtAOBuffer[i]=ExtFastBuffer[i]-ExtSlowBuffer[i];
     
   //--- copy correct data into the buffer
   ArrayCopy(AO, ExtAOBuffer, 0, 200, ArraySize(ExtAOBuffer) - 200 - 1);
   ArrayReverse(AO, 0, ArraySize(AO));
}

//+--------------------------------------------------------------------------------------------------------------+

void getUO(int InpFastPeriod, int InpMiddlePeriod, int InpSlowPeriod, double &UO[], int InpFastK, int InpMiddleK, int InpSlowK, MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(UO) = ArraySize(RatesSedonde) - 200 - 1.     //"-200":precision margin ; "-1":starts at the second last
     
     UO will be of the TimeSeries form : UO[0] is the most recent value.
     
     //--- default parameters
     InpFastPeriod=7;     // Fast ATR period
     InpMiddlePeriod=14;  // Middle ATR period
     InpSlowPeriod=28;    // Slow ATR period
     InpFastK=4;          // Fast K
     InpMiddleK=2;        // Middle K
     InpSlowK=1;          // Slow K
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200)
   {
      printf("Erreur(indicators.mqh:getUO) : size of RatesSeconde > 200.");
      ArrayInitialize(UO, 0.0);
      return;
   }
   
   if(InpSlowPeriod<InpMiddlePeriod || InpMiddlePeriod<InpFastPeriod)
   {
      printf("Erreur(indicators.mqh:getUO) : InpSlowPeriod < InpMiddlePeriod < InpFastPeriod");
      ArrayInitialize(UO, 0.0);
      return;
   }

   //--- indicator buffers
   double     ExtUOBuffer[];
   ArrayResize(ExtUOBuffer,rates_total, 0);
   double     ExtBPBuffer[];
   ArrayResize(ExtBPBuffer,rates_total, 0);
   double     ExtFastATRBuffer[];
   ArrayResize(ExtFastATRBuffer,rates_total, 0);
   double     ExtMiddleATRBuffer[];
   ArrayResize(ExtMiddleATRBuffer,rates_total, 0);
   double     ExtSlowATRBuffer[];
   ArrayResize(ExtSlowATRBuffer,rates_total, 0);
   
   //--- indicator handle
   double    ExtDivider;
   
   //--- price buffer
   double    close[], low[];
   ArrayResize(close, rates_total, 0);
   ArrayResize(low, rates_total, 0);
   for(int i=0; i<rates_total; i++)
   {
      close[rates_total-1-i] = RatesSeconde[i].close;
      low[rates_total-1-i] = RatesSeconde[i].low;      
   }
   
   //--- get Buffer
   getATR(InpFastPeriod,ExtFastATRBuffer,RatesSeconde);
   getATR(InpMiddlePeriod,ExtMiddleATRBuffer,RatesSeconde);
   getATR(InpSlowPeriod,ExtSlowATRBuffer,RatesSeconde);
   
   ExtDivider=InpFastK+InpMiddleK+InpSlowK;
     
   //--- preliminary calculations
   ExtBPBuffer[0]=0.0;
   ExtUOBuffer[0]=0.0;
   //--- set value for first InpSlowPeriod bars
   for(int i=1; i<=InpSlowPeriod; i++)
     {
      ExtUOBuffer[i]=0.0;
      double true_low=MathMin(low[i],close[i]);
      ExtBPBuffer[i]=close[i]-true_low;
     }
   
   //--- the main loop of calculations
   for(int i=InpSlowPeriod+1; i<rates_total; i++)
     {
      double true_low=MathMin(low[i],close[i]);
      ExtBPBuffer[i]=close[i]-true_low;            // buying pressure

      if(ExtFastATRBuffer[i]!=0.0 &&
         ExtMiddleATRBuffer[i]!=0.0 &&
         ExtSlowATRBuffer[i]!=0.0)
        {
         double raw_uo=InpFastK*SimpleMA(i,InpFastPeriod,ExtBPBuffer)/ExtFastATRBuffer[i]+
                       InpMiddleK*SimpleMA(i,InpMiddlePeriod,ExtBPBuffer)/ExtMiddleATRBuffer[i]+
                       InpSlowK*SimpleMA(i,InpSlowPeriod,ExtBPBuffer)/ExtSlowATRBuffer[i];
         ExtUOBuffer[i]=raw_uo/ExtDivider*100.0;
        }
      else
         ExtUOBuffer[i]=ExtUOBuffer[i-1]; // set current Ultimate value as previous Ultimate value
     }
     
   //--- copy correct data into the buffer
   ArrayCopy(UO, ExtUOBuffer, 0, 200, ArraySize(ExtUOBuffer) - 200 - 1);
   ArrayReverse(UO, 0, ArraySize(UO));
}

//+--------------------------------------------------------------------------------------------------------------+

void getATR(int ExtPeriodATR, double &ATR[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(ATR) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     ATR will be of the TimeSeries form : ATR[0] is the most recent value.
     
     //--- default parameters
     ExtPeriodATR=14;     // ATR period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getATR) : size of RatesSecond > 200.");
      ArrayInitialize(ATR, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtATRBuffer[];
   ArrayResize(ExtATRBuffer,rates_total, 0);
   double    ExtTRBuffer[];
   ArrayResize(ExtTRBuffer,rates_total, 0);
   
   //--- price buffer
   double    close[], low[], high[];
   ArrayResize(close, rates_total, 0);
   ArrayResize(low, rates_total, 0);
   ArrayResize(high, rates_total, 0);
   
   for(int i=0; i<rates_total; i++)
   {
      close[rates_total-1-i] = RatesSeconde[i].close;
      low[rates_total-1-i] = RatesSeconde[i].low;
      high[rates_total-1-i] =  RatesSeconde[i].high;
   }
   
   //--- preliminary calculations
   ExtTRBuffer[0]=0.0;
   ExtATRBuffer[0]=0.0;
   
   //--- filling out the array of True Range values for each period
   for(int i=1; i<rates_total; i++) ExtTRBuffer[i]=MathMax(high[i],close[i-1])-MathMin(low[i],close[i-1]);
      
   //--- first AtrPeriod values of the indicator are not calculated
   double firstValue=0.0;
   for(int i=1; i<=ExtPeriodATR; i++)
     {
      ExtATRBuffer[i]=0.0;
      firstValue+=ExtTRBuffer[i];
     }
   //--- calculating the first value of the indicator
   firstValue/=ExtPeriodATR;
   ExtATRBuffer[ExtPeriodATR]=firstValue;

   //--- the main loop of calculations
   for(int i=ExtPeriodATR+1; i<rates_total; i++)
   {
      ExtTRBuffer[i]=MathMax(high[i],close[i-1])-MathMin(low[i],close[i-1]);
      ExtATRBuffer[i]=ExtATRBuffer[i-1]+(ExtTRBuffer[i]-ExtTRBuffer[i-ExtPeriodATR])/ExtPeriodATR;
   }

   //--- copy correct data into the buffer
   ArrayCopy(ATR, ExtATRBuffer, 0, 200, ArraySize(ExtATRBuffer) - 200 - 1);
   ArrayReverse(ATR, 0, ArraySize(ATR));
}

//+--------------------------------------------------------------------------------------------------------------+

void getRVI(int InpRVIPeriod, int TRIANGLE_PERIOD, double &RVI[], double &SIGNAL[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(RVI) = ArraySize(RatesSedonde) - 200 - 1.    //"-200":precision margin ; "-1":starts at the second last
     
     RVI will be of the TimeSeries form : RVI[0] is the most recent value (excludes the current candle).
     
     //--- default parameters
     InpRVIPeriod = 10;     // RVI period
     TRIANGLE_PERIOD = 3;
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- input parameters
   int AVERAGE_PERIOD = (TRIANGLE_PERIOD*2);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getRVI) : size of RatesSecond > 200.");
      ArrayInitialize(RVI, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtRVIBuffer[];
   double    ExtSignalBuffer[];
   ArrayResize(ExtRVIBuffer,rates_total, 0);
   ArrayResize(ExtSignalBuffer,rates_total, 0);
   
   //--- price buffer
   double    close[], low[], high[], open[];
   ArrayResize(close, rates_total, 0);
   ArrayResize(low, rates_total, 0);
   ArrayResize(high, rates_total, 0);
   ArrayResize(open, rates_total, 0);
   
   for(int i=0; i<rates_total; i++)
   {
      close[rates_total-1-i] = RatesSeconde[i].close;
      low[rates_total-1-i] = RatesSeconde[i].low;
      high[rates_total-1-i] =  RatesSeconde[i].high;
      open[rates_total-1-i] = RatesSeconde[i].open;
   }
   
   //--- set empty value for uncalculated bars
   for(int i=0; i<InpRVIPeriod+TRIANGLE_PERIOD; i++)
         ExtRVIBuffer[i]=0.0;
   for(int i=0; i<InpRVIPeriod+AVERAGE_PERIOD; i++)
         ExtSignalBuffer[i]=0.0;
   
   //--- RVI counted in the 1-st buffer
   double sum_up, sum_down, value_up, value_down;
   for(int i=InpRVIPeriod+2; i<rates_total; i++)
     {
      sum_up=0.0;
      sum_down=0.0;
      for(int j=i; j>i-InpRVIPeriod; j--)
        {
         value_up=close[j]-open[j]+2*(close[j-1]-open[j-1])+2*(close[j-2]-open[j-2])+close[j-3]-open[j-3];
         value_down=high[j]-low[j]+2*(high[j-1]-low[j-1])+2*(high[j-2]-low[j-2])+high[j-3]-low[j-3];
         sum_up+=value_up;
         sum_down+=value_down;
        }
      if(sum_down!=0.0)
         ExtRVIBuffer[i]=sum_up/sum_down;
      else
         ExtRVIBuffer[i]=sum_up;
     }
   //--- signal line counted in the 2-nd buffer
   int start = InpRVIPeriod + TRIANGLE_PERIOD + 2;
   for(int i=start; i<rates_total; i++) ExtSignalBuffer[i]=(ExtRVIBuffer[i] + 2*ExtRVIBuffer[i-1] + 2*ExtRVIBuffer[i-2] + ExtRVIBuffer[i-3])/AVERAGE_PERIOD;
   
   //--- copy correct data into the buffer
   ArrayCopy(RVI, ExtRVIBuffer, 0, 200, ArraySize(ExtRVIBuffer) - 200 - 1);
   ArrayReverse(RVI, 0, ArraySize(RVI));
   
   ArrayCopy(SIGNAL, ExtSignalBuffer, 0, 200, ArraySize(ExtRVIBuffer) - 200 - 1);
   ArrayReverse(SIGNAL, 0, ArraySize(SIGNAL));
}

//+--------------------------------------------------------------------------------------------------------------+

void getSTOCH(int InpKPeriod, int InpDPeriod, int InpSlowing, double &STOCH[], double &SIGNAL[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(STOCH) = ArraySize(RatesSedonde) - 200 - 1. //"-200":precision margin ; "-1":starts at the second last
     
     STOCH will be of the TimeSeries form : STOCH[0] is the most recent value (excludes the current candle).
     
     //--- default parameters
     input int InpKPeriod=5;  // K period
     input int InpDPeriod=3;  // D period
     input int InpSlowing=3;  // Slowing
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getSTOCH) : size of RatesSecond > 200.");
      ArrayInitialize(STOCH, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtMainBuffer[];
   double    ExtSignalBuffer[];
   double    ExtHighesBuffer[];
   double    ExtLowesBuffer[];
   ArrayResize(ExtMainBuffer,rates_total, 0);
   ArrayResize(ExtSignalBuffer,rates_total, 0);
   ArrayResize(ExtHighesBuffer,rates_total, 0);
   ArrayResize(ExtLowesBuffer,rates_total, 0);
   
   //--- price buffer
   double    low[], high[], close[];
   ArrayResize(low, rates_total, 0);
   ArrayResize(high, rates_total, 0);
   ArrayResize(close, rates_total, 0);
   
   for(int i=0; i<rates_total; i++)
   {
      low[rates_total-1-i] = RatesSeconde[i].low;
      high[rates_total-1-i] =  RatesSeconde[i].high;
      close[rates_total-1-i] =  RatesSeconde[i].close;
   }
   
   for(int i=0; i<InpKPeriod-1; i++)
   {
      ExtLowesBuffer[i]=0.0;
      ExtHighesBuffer[i]=0.0;
   }
   
   //--- calculate HighesBuffer[] and ExtHighesBuffer[]
   double dmin, dmax;
   for(int i=InpKPeriod-1; i<rates_total; i++)
     {
      dmin=1000000.0;
      dmax=-1000000.0;
      for(int k=i-InpKPeriod+1; k<=i; k++)
        {
         if(dmin>low[k])
            dmin=low[k];
         if(dmax<high[k])
            dmax=high[k];
        }
      ExtLowesBuffer[i]=dmin;
      ExtHighesBuffer[i]=dmax;
     }
   //--- %K
   int start=InpKPeriod-1+InpSlowing-1;
   for(int i=0; i<start; i++) ExtMainBuffer[i]=0.0;
      
   //--- main cycle
   for(int i=start; i<rates_total; i++)
     {
      double sum_low=0.0;
      double sum_high=0.0;
      for(int k=(i-InpSlowing+1); k<=i; k++)
        {
         sum_low +=(close[k]-ExtLowesBuffer[k]);
         sum_high+=(ExtHighesBuffer[k]-ExtLowesBuffer[k]);
        }
      if(sum_high==0.0)
         ExtMainBuffer[i]=100.0;
      else
         ExtMainBuffer[i]=sum_low/sum_high*100;
     }
   //--- signal
   start=InpDPeriod-1;
   for(int i=0; i<start; i++) ExtSignalBuffer[i]=0.0;
   
   double sum;
   for(int i=start; i<rates_total; i++)
     {
      sum=0.0;
      for(int k=0; k<InpDPeriod; k++) sum+=ExtMainBuffer[i-k];
      ExtSignalBuffer[i]=sum/InpDPeriod;
     }
     
   //--- copy correct data into the buffer
   ArrayCopy(STOCH, ExtMainBuffer, 0, 200, ArraySize(ExtMainBuffer) - 200 - 1);
   ArrayReverse(STOCH, 0, ArraySize(STOCH));
   
   ArrayCopy(SIGNAL, ExtSignalBuffer, 0, 200, ArraySize(ExtMainBuffer) - 200 - 1);
   ArrayReverse(SIGNAL, 0, ArraySize(SIGNAL));
}

//+--------------------------------------------------------------------------------------------------------------+

void getBearsBullsPower(int InpBearsPeriod, int InpBullsPeriod, double &Bears[], double &Bulls[], double &Variations[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(Bears) = ArraySize(RatesSedonde) - 200 - 1.        //"-200":precision margin ; "-1":starts at the second last
     ArraySize(Bulls) = ArraySize(RatesSedonde) - 200 - 1.        //"-200":precision margin ; "-1":starts at the second last
     ArraySize(Variations) = ArraySize(RatesSedonde) - 200 - 1.   //"-200":precision margin ; "-1":starts at the second last
     
     BearsBulls will be of the TimeSeries form : BearsBulls[0] is the most recent value (excludes the current candle).
     
     //--- default parameters
     input int InpBearsPeriod=13;
     input int InpBullsPeriod=13;
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getBearsBullsPower) : size of RatesSecond > 200.");
      ArrayInitialize(Bears, 0.0);
      ArrayInitialize(Bulls, 0.0);
      ArrayInitialize(Variations, 0.0);
      return;
   }
   
   //--- price buffer
   double    close[], low[], high[];
   ArrayResize(close, rates_total, 0);
   ArrayResize(low, rates_total, 0);
   ArrayResize(high, rates_total, 0);
   for(int i=0; i<rates_total; i++)
   {
      low[rates_total-1-i] = RatesSeconde[i].low;
      high[rates_total-1-i] =  RatesSeconde[i].high;
      close[rates_total-1-i] = RatesSeconde[i].close;
   }

   //--- indicator buffers
   double    ExtBearsBuffer[];
   double    ExtBullsBuffer[];
   double    VariationsBearsBulls[];
   double    ExtTempBuffer_Bears[];
   double    ExtTempBuffer_Bulls[];
   ArrayResize(ExtBearsBuffer,rates_total, 0);
   ArrayResize(ExtBullsBuffer,rates_total, 0);
   ArrayResize(VariationsBearsBulls,rates_total, 0);
   ArrayResize(ExtTempBuffer_Bears,rates_total, 0);
   ArrayResize(ExtTempBuffer_Bulls,rates_total, 0);
   double    TempVarBuffer[]; //temporary buffer for the use of the getEMA function
   
   getEMA(InpBearsPeriod,0,ExtTempBuffer_Bears,TempVarBuffer,close);
   getEMA(InpBullsPeriod,0,ExtTempBuffer_Bulls,TempVarBuffer,close);
   
   //--- the main loop of calculations
   for(int i=InpBearsPeriod; i<rates_total; i++) ExtBearsBuffer[i]=low[i]-ExtTempBuffer_Bears[i];
   for(int i=InpBullsPeriod; i<rates_total; i++)
   {
      ExtBullsBuffer[i]=high[i]-ExtTempBuffer_Bulls[i];
      VariationsBearsBulls[i] = 100*(ExtBullsBuffer[i] - ExtBearsBuffer[i])/ExtBearsBuffer[i];
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(Bears, ExtBearsBuffer, 0, 200, ArraySize(ExtBearsBuffer) - 200 - 1);
   ArrayReverse(Bears, 0, ArraySize(Bears));
   
   ArrayCopy(Bulls, ExtBullsBuffer, 0, 200 , ArraySize(ExtBullsBuffer) - 200 - 1);
   ArrayReverse(Bulls, 0, ArraySize(Bulls));
   
   ArrayCopy(Variations, VariationsBearsBulls, 0, 200, ArraySize(VariationsBearsBulls) - 200 - 1);
   ArrayReverse(Variations, 0, ArraySize(Variations));
}

//+--------------------------------------------------------------------------------------------------------------+

void getEMA(int InpPeriodEMA, int InpShift, double &EMA[], double &Variation[], double &close[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(EMA) = ArraySize(RatesSedonde) - 200 - 1.       //"-200":precision margin ; "-1":starts at the second last
     ArraySize(Variation) = ArraySize(RatesSedonde) - 200 - 1. //"-200":precision margin ; "-1":starts at the second last
     
     EMA will be of the TimeSeries form : EMA[0] is the most recent value (excludes the current candle).
     
     //--- default parameters
     int InpPeriodEMA = 14;
   *************************************************************************************************************/
   
   int rates_total = ArraySize(close);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getEma) : size of RatesSecond > 200.");
      ArrayInitialize(EMA, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtEmaBuffer[];
   ArrayResize(ExtEmaBuffer,rates_total, 0);
   double    VariationEMABuffer[];
   ArrayResize(VariationEMABuffer,rates_total, 0);

   //--- calculate start position
   double smooth_factor=2.0/(1.0+InpPeriodEMA);
   ExtEmaBuffer[0] = close[0];
   
   //--- main loop
   for(int i=1; i<rates_total; i++)
   {
      ExtEmaBuffer[i] = close[i]*smooth_factor + ExtEmaBuffer[i-1]*(1.0-smooth_factor);
      VariationEMABuffer[i] = 100*(close[i] - ExtEmaBuffer[i])/ExtEmaBuffer[i];
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(EMA, ExtEmaBuffer, 0, 200, ArraySize(ExtEmaBuffer) - 200 - 1);
   ArrayReverse(EMA, 0, ArraySize(EMA));
   ArrayCopy(Variation, VariationEMABuffer, 0, 200, ArraySize(VariationEMABuffer) - 200 - 1);
   ArrayReverse(Variation, 0, ArraySize(Variation));
}

//+--------------------------------------------------------------------------------------------------------------+

void getDEMA(int InpPeriodDEMA, int InpShift, double &DEMA[], double &Variation[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(DEMA) = ArraySize(RatesSedonde) - 200 - 1.      //"-200":precision margin ; "-1":starts at the second last
     ArraySize(Variation) = ArraySize(RatesSedonde) - 200 - 1. //"-200":precision margin ; "-1":starts at the second last
     
     Dema will be of the TimeSeries form : DEMA[0] is the most recent value (second last value).
     
     //--- default parameters
     int InpPeriodDEMA = 14;
     int InpShift=0;        // Indicator's shift
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- price buffer
   double    close[];
   ArrayResize(close, rates_total, 0);
   for(int i=0; i<rates_total; i++) close[rates_total-1-i] = RatesSeconde[i].close;
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getDema) : size of RatesSecond > 200.");
      ArrayInitialize(DEMA, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtDemaBuffer[];
   double    ExtEmaBuffer[];
   double    ExtEmaOfEmaBuffer[];
   double    VariationDEMABuffer[];
   ArrayResize(VariationDEMABuffer,rates_total, 0);
   ArrayResize(ExtDemaBuffer,rates_total, 0);
   ArrayResize(ExtEmaBuffer,rates_total, 0);
   ArrayResize(ExtEmaOfEmaBuffer,rates_total, 0);
   
   //--- calculate EMA
   double VarEma[];
   getEMA(InpPeriodDEMA,InpShift,ExtEmaBuffer, VarEma,close);
   ArrayReverse(ExtEmaBuffer, 0, ArraySize(ExtEmaBuffer));
   
   //--- calculate EMA on EMA array
   double ExtEmaBufferLessPeriod[];
   ArrayResize(ExtEmaBufferLessPeriod,rates_total-InpPeriodDEMA, 0);
   for(int i=InpPeriodDEMA; i<rates_total; i++) ExtEmaBufferLessPeriod[i-InpPeriodDEMA] = ExtEmaBuffer[i];
   
   getEMA(InpPeriodDEMA,InpShift,ExtEmaOfEmaBuffer, VarEma, ExtEmaBufferLessPeriod);
   ArrayReverse(ExtEmaOfEmaBuffer, 0, ArraySize(ExtEmaOfEmaBuffer));
   
   //--- calculate DEMA
   for(int i=0; i<InpPeriodDEMA; i++)
   {
      ExtDemaBuffer[i]=2.0*ExtEmaBuffer[i];
      VariationDEMABuffer[i] = 100*(close[i] - ExtDemaBuffer[i])/ExtDemaBuffer[i];
   }
   for(int i=InpPeriodDEMA; i<rates_total; i++)
   {
      ExtDemaBuffer[i]=2.0*ExtEmaBuffer[i] - ExtEmaOfEmaBuffer[i-InpPeriodDEMA];
      if(ExtDemaBuffer[i]==0) VariationDEMABuffer[i]=100;
      else VariationDEMABuffer[i] = 100*(close[i] - ExtDemaBuffer[i])/ExtDemaBuffer[i];
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(DEMA, ExtDemaBuffer, 0, 200, ArraySize(ExtDemaBuffer) - 200 - 1);                   //"-200":precision margin ; "-1":starts at the second last
   ArrayReverse(DEMA, 0, ArraySize(DEMA));
   ArrayCopy(Variation, VariationDEMABuffer, 0, 200, ArraySize(VariationDEMABuffer) - 200  - 1); //"-200":precision margin ; "-1":starts at the second last
   ArrayReverse(Variation, 0, ArraySize(Variation));
}

//+--------------------------------------------------------------------------------------------------------------+

void getSMA(int InpPeriodSMA, int InpShift, double &SMA[], double &Variation[], double &close[])
{
   /*************************************************************************************************************
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(SMA) = ArraySize(RatesSedonde) - 200 - 1.        //"-200":precision margin ; "-1":starts at the second last
     ArraySize(Variation) = ArraySize(RatesSedonde) - 200 - 1.  //"-200":precision margin ; "-1":starts at the second last
     
     SMA will be of the TimeSeries form : SMA[0] is the most recent value (second last value).
     
     //--- default parameters
     int InpPeriodSMA = 14;
   *************************************************************************************************************/
   
   int rates_total = ArraySize(close);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getSMA) : size of RatesSecond > 200.");
      ArrayInitialize(SMA, 0.0);
      return;
   }

   //--- indicator buffers
   double    ExtSMABuffer[];
   ArrayResize(ExtSMABuffer,rates_total, 0);
   double    VariationSMABuffer[];
   ArrayResize(VariationSMABuffer,rates_total, 0);
   double first_value=0;

   //--- set empty value for first bars
   for(int i=0; i<InpPeriodSMA; i++)
   {
      ExtSMABuffer[i]=0.0;
      VariationSMABuffer[i]=0.0;
      first_value+=close[i];
   }
   ExtSMABuffer[InpPeriodSMA-1]=first_value/InpPeriodSMA;
   VariationSMABuffer[InpPeriodSMA-1] = 100*(close[InpPeriodSMA-1] - ExtSMABuffer[InpPeriodSMA-1])/ExtSMABuffer[InpPeriodSMA-1];

   //--- main loop
   for(int i=InpPeriodSMA; i<rates_total; i++)
   {
      ExtSMABuffer[i] = ExtSMABuffer[i-1] + (close[i] - close[i-InpPeriodSMA])/InpPeriodSMA;
      VariationSMABuffer[i] = 100*(close[i] - ExtSMABuffer[i])/ExtSMABuffer[i];
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(SMA, ExtSMABuffer, 0, 0, ArraySize(ExtSMABuffer) -1);                     //"-200":precision margin ; "-1":starts at the second last
   ArrayReverse(SMA, 0, ArraySize(SMA));
   ArrayCopy(Variation, VariationSMABuffer, 0, 0, ArraySize(VariationSMABuffer) -1);   //"-200":precision margin ; "-1":starts at the second last
   ArrayReverse(Variation, 0, ArraySize(Variation));
}

//+--------------------------------------------------------------------------------------------------------------+

void getMACD(int InpFastEMA, int InpSlowEMA, int InpSignalSMA, double &MACD[], double &Signal[], MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     Add 200 + InpSignalSMA to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(MACD) = ArraySize(RatesSedonde) - 200 - InpSignalSMA.    //"-200":precision margin ; "-InpSignalSMA":starts at the second last & due to signal calculation
     ArraySize(Signal) = ArraySize(RatesSedonde) - 200 - InpSignalSMA.  //"-200":precision margin ; "-InpSignalSMA":starts at the second last & due to signal calculation
     
     MACD will be of the TimeSeries form : MACD[0] is the most recent value (second last value).
     
     //--- default parameters
     int InpFastEMA=12;   // Fast EMA period
     int InpSlowEMA=26;   // Slow EMA period
     int InpSignalSMA=9;  // Signal SMA period
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200){
   
      printf("Erreur(indicators.mqh:getMACD) : size of RatesSecond > 200.");
      ArrayInitialize(MACD, 0.0);
      return;
   }
   
   //--- price buffer
   double    close[];
   ArrayResize(close, rates_total, 0);
   for(int i=0; i<rates_total; i++) close[rates_total-1-i] = RatesSeconde[i].close;

   //--- indicator buffers
   double ExtMacdBuffer[];
   double ExtSignalBuffer[];
   double ExtFastMaBuffer[];
   double ExtSlowMaBuffer[];
   ArrayResize(ExtMacdBuffer,rates_total, 0);
   ArrayResize(ExtSignalBuffer,rates_total, 0);
   ArrayResize(ExtFastMaBuffer,rates_total, 0);
   ArrayResize(ExtSlowMaBuffer,rates_total, 0);
   
   //--- get MA buffers
   double TempVarBuffer[];    //temporary buffer for the use of the getEMA function
   getEMA(InpFastEMA,0,ExtFastMaBuffer,TempVarBuffer,close);
   getEMA(InpSlowEMA,0,ExtSlowMaBuffer,TempVarBuffer,close);
   
   //--- calculate MACD
   for(int i=0; i<rates_total; i++) ExtMacdBuffer[i]=ExtFastMaBuffer[i]-ExtSlowMaBuffer[i];
   
   //--- calculate Signal
   getSMA(InpSignalSMA,0,ExtSignalBuffer,TempVarBuffer,ExtMacdBuffer);
   
   //--- copy correct data into the buffer
   ArrayCopy(MACD, ExtMacdBuffer, 0, 200+1, ArraySize(ExtMacdBuffer) - 200);
   ArrayCopy(Signal, ExtSignalBuffer, 0, 200+1, ArraySize(ExtSignalBuffer) - 200);
   ArrayReverse(Signal, 0, ArraySize(Signal));
}

//+--------------------------------------------------------------------------------------------------------------+

void getICHIMOKU(int InpTenkan,
                 int InpKijun,
                 int InpSenkou, 
                 double &TENKAN[], 
                 double &KIJUN[], 
                 double &SSB[], 
                 double &SSA[], 
                 double &VariationTenkan[], 
                 double &VariationKijun[], 
                 double &VariationSSB[], 
                 double &VariationSSA[], 
                 double &VariationSSB_SSA[], 
                 MyMqlRates &RatesSeconde[])
{
   /*************************************************************************************************************
     We have chosen not to use the chikou as it is only a copy of the price in the past.
   
     Add 200 to the size of the RatesSecond array (= the number of desired data) to get the correct data.
     
     ArraySize(TENKAN) = ArraySize(RatesSedonde) - 200 - 1.                //"-200":precision margin ; "-1":starts at the second last
     ArraySize(KIJUN) = ArraySize(RatesSedonde) - 200 - 1.                 //"-200":precision margin ; "-1":starts at the second last
     ArraySize(SSB) = ArraySize(RatesSedonde) - 200 - 1.                   //"-200":precision margin ; "-1":starts at the second last
     ArraySize(SSA) = ArraySize(RatesSedonde) - 200 - 1.                   //"-200":precision margin ; "-1":starts at the second last
     ArraySize(VariationTenkan) = ArraySize(RatesSedonde) - 200 - 1.       //"-200":precision margin ; "-1":starts at the second last
     ArraySize(VariationKijun) = ArraySize(RatesSedonde) - 200 - 1.        //"-200":precision margin ; "-1":starts at the second last
     ArraySize(VariationSSB) = ArraySize(RatesSedonde) - 200 - 1.          //"-200":precision margin ; "-1":starts at the second last
     ArraySize(VariationSSA) = ArraySize(RatesSedonde) - 200 - 1.          //"-200":precision margin ; "-1":starts at the second last
     ArraySize(VariationSSB_SSA) = ArraySize(RatesSedonde) - 200 - 1.      //"-200":precision margin ; "-1":starts at the second last
     
     TENKAN, KIJUN, SSB, SSA, VariationTenkan, VariationKijun, VariationSSB, VariationSSA, VariationSSB_SSA
     will be of the TimeSeries form : TENKAN[0] is the most recent value (second last value) for exemple.
     
     //--- default parameters
     int InpTenkan=9;     // Tenkan-sen
     int InpKijun=26;     // Kijun-sen
     int InpSenkou=52;    // Senkou Span B
   *************************************************************************************************************/
   
   int rates_total = ArraySize(RatesSeconde);
   
   //--- check for input value of period
   if(rates_total <= 200 || InpTenkan>InpKijun || InpKijun>InpSenkou){
   
      printf("Erreur(indicators.mqh:getICHIMOKU) : size of RatesSecond > 200.");
      ArrayInitialize(TENKAN, 0.0);
      ArrayInitialize(KIJUN, 0.0);
      ArrayInitialize(SSB, 0.0);
      ArrayInitialize(SSA, 0.0);
      
      ArrayInitialize(VariationTenkan, 0.0);
      ArrayInitialize(VariationKijun, 0.0);
      ArrayInitialize(VariationSSB, 0.0);
      ArrayInitialize(VariationSSA, 0.0);
      ArrayInitialize(VariationSSB_SSA, 0.0);
      return;
   }
   
   //--- price buffer
   double    close[], low[], high[];
   ArrayResize(close, rates_total, 0);
   ArrayResize(low, rates_total, 0);
   ArrayResize(high, rates_total, 0);
   
   for(int i=0; i<rates_total; i++)
   {
      close[rates_total-1-i] = RatesSeconde[i].close;
      low[rates_total-1-i] = RatesSeconde[i].low;
      high[rates_total-1-i] =  RatesSeconde[i].high;
   }

   //--- indicator buffers
   double    ExtTenkanBuffer[];
   double    ExtKijunBuffer[];
   double    ExtSpanABuffer[];
   double    ExtSpanBBuffer[];
   ArrayResize(ExtTenkanBuffer,rates_total-InpSenkou, 0);
   ArrayResize(ExtKijunBuffer,rates_total-InpSenkou, 0);
   ArrayResize(ExtSpanABuffer,rates_total-InpSenkou, 0);
   ArrayResize(ExtSpanBBuffer,rates_total-InpSenkou, 0);
   
   //--- variation buffers
   double VarTenkan[], VarKijun[], VarSSB[], VarSSA[], VarSSB_SSA[];
   ArrayResize(VarTenkan, rates_total-InpSenkou, 0);
   ArrayResize(VarKijun, rates_total-InpSenkou, 0);
   ArrayResize(VarSSB, rates_total-InpSenkou, 0);
   ArrayResize(VarSSA, rates_total-InpSenkou, 0);
   ArrayResize(VarSSB_SSA, rates_total-InpSenkou, 0);
   
   //--- main loop
   double price_max, price_min;
   for(int i=InpSenkou; i<rates_total; i++)
   {
      //--- tenkan sen
      price_max=high[i];
      price_min=low[i];
      for(int j=i; j>i-InpTenkan; j--)
      {
         if(price_max<high[j]) price_max = high[j];
         if(price_min>low[j]) price_min = low[j];
      }
      ExtTenkanBuffer[i-InpSenkou]=(price_max+price_min)/2.0;
      VarTenkan[i-InpSenkou]=100.0*(ExtTenkanBuffer[i-InpSenkou] - close[i])/close[i];
      
      //--- kijun sen
      for(int j=i; j>i-InpKijun; j--)
      {
         if(price_max<high[j]) price_max = high[j];
         if(price_min>low[j]) price_min = low[j];
      }
      ExtKijunBuffer[i-InpSenkou]=(price_max+price_min)/2.0;
      VarKijun[i-InpSenkou]=100.0*(ExtKijunBuffer[i-InpSenkou] - close[i])/close[i];
      
      //--- senkou span a
      ExtSpanABuffer[i-InpSenkou]=(ExtTenkanBuffer[i-InpSenkou]+ExtKijunBuffer[i-InpSenkou])/2.0;
      VarSSA[i-InpSenkou]=100.0*(ExtSpanABuffer[i-InpSenkou] - close[i])/close[i];
      
      //--- senkou span b
      for(int j=i; j>i-InpSenkou; j--)
      {
         if(price_max<high[j]) price_max = high[j];
         if(price_min>low[j]) price_min = low[j];
      }
      ExtSpanBBuffer[i-InpSenkou]=(price_max+price_min)/2.0;
      VarSSB[i-InpSenkou]=100.0*(ExtSpanBBuffer[i-InpSenkou] - close[i])/close[i];
      
      VarSSB_SSA[i-InpSenkou] = 100.0*(ExtSpanABuffer[i-InpSenkou] - ExtSpanBBuffer[i-InpSenkou])/ExtSpanBBuffer[i-InpSenkou];
   }
   
   //--- copy correct data into the buffer
   ArrayCopy(TENKAN, ExtTenkanBuffer, 0, 200-InpSenkou, ArraySize(ExtTenkanBuffer) - 200+InpSenkou - 1);
   ArrayReverse(TENKAN, 0, ArraySize(TENKAN));
   ArrayCopy(KIJUN, ExtKijunBuffer, 0, 200-InpSenkou, ArraySize(ExtKijunBuffer) - 200+InpSenkou - 1);
   ArrayReverse(KIJUN, 0, ArraySize(KIJUN));
   ArrayCopy(SSB, ExtSpanBBuffer, 0, 200-InpKijun-InpSenkou, ArraySize(ExtSpanBBuffer) - 200+InpSenkou - 1);
   ArrayReverse(SSB, 0, ArraySize(SSB));
   ArrayCopy(SSA, ExtSpanABuffer, 0, 200-InpKijun-InpSenkou, ArraySize(ExtSpanABuffer) - 200+InpSenkou - 1);
   ArrayReverse(SSA, 0, ArraySize(SSA));
   
   ArrayCopy(VariationTenkan, VarTenkan, 0, 200-InpSenkou, ArraySize(VarTenkan) - 200+InpSenkou - 1);
   ArrayReverse(VariationTenkan, 0, ArraySize(VariationTenkan));
   ArrayCopy(VariationKijun, VarKijun, 0, 200-InpSenkou, ArraySize(VarKijun) - 200+InpSenkou - 1);
   ArrayReverse(VariationKijun, 0, ArraySize(VariationKijun));
   ArrayCopy(VariationSSB, VarSSB, 0, 200-InpSenkou, ArraySize(VarSSB) - 200+InpSenkou - 1);
   ArrayReverse(VariationSSB, 0, ArraySize(VariationSSB));
   ArrayCopy(VariationSSA, VarSSA, 0, 200-InpSenkou, ArraySize(VarSSA) - 200+InpSenkou - 1);
   ArrayReverse(VariationSSA, 0, ArraySize(VariationSSA));
   ArrayCopy(VariationSSB_SSA, VarSSB_SSA, 0, 200-InpSenkou, ArraySize(VarSSB_SSA) - 200+InpSenkou - 1);
   ArrayReverse(VariationSSB_SSA, 0, ArraySize(VariationSSB_SSA));
}