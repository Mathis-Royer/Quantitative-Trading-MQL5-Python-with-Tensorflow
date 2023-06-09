//+------------------------------------------------------------------+
//|                                                  market_data.mqh |
//|                                       Copyright 2022, Hedge Ltd. |
//|                                            https://www.hedge.com |
//+------------------------------------------------------------------+
#property copyright     "Copyright 2022, Hedge Ltd."
#property link          "https://www.hedge.com"
#property description   "10 seconds Market data"
#property version       "1.00"
//+--------------------------------------------------------------------------------------------------------------+
#include "structure.mqh"
//+--------------------------------------------------------------------------------------------------------------+

void getValideTimerange(datetime &time_current)
{
   //--- Calcul of FromPast datetime (2min before)
   MqlDateTime daycurrent;
   TimeToStruct(time_current, daycurrent);
   int hours = daycurrent.hour;
   int minutes = daycurrent.min;
   if(minutes<=1)
   {
      hours = daycurrent.hour - 1;
      minutes = 60 - 1 + daycurrent.min;
   }
   else minutes -= 1;
   time_current = StringToTime((string)daycurrent.year + "." + (string)daycurrent.mon + "." + (string)daycurrent.day + " " + (string)hours + ":" + (string)minutes + ":" + "00");
   
}


int getRatesSeconde(string symbol, datetime From_Past, datetime To_Present, MyMqlRates &Rates_Seconde[], bool t, bool o, bool h, bool l, bool c, bool tv, bool s, bool tch, bool tcl, bool vco, bool vch, bool vcl,bool ap)
{
   /*********************************************************************
     Modifie le tableau rates_array avec les valeurs de rates (close,
     high, low, open, real_volume, spread, tick_volume, time) des bougies
     entre la start_pos ième bougie et la (start_pos + count) ième bougie
     par rapport à la bougie actuelle
   ______________________________________________________
   ...| | | | | | | | | | | | | | | | | | | | | | | | | |
         ^                                           ^ ^
         |_(start_pos + count)           (start_pos)_| |_(Actual position)
   **********************************************************************/
   //uint start_pos = Bars(symbol,PERIOD_M1,To_Present,TimeCurrent());    //counts the number of bars 
   
   getValideTimerange(To_Present);
   
   ulong nb_bars = ((ulong)To_Present - (ulong)From_Past)/60-1;
   int count = Bars(symbol,PERIOD_M1,From_Past,To_Present);            //counts the number of bars
   if(count < (int)nb_bars){printf("<!>ERROR Bars<!> : Number of bars downloaded : %d. Number of bars required : %d", count, nb_bars); return 0;}
   
   
   //printf("count = %d : %s", count, symbol);
   
   MqlTick ticks[];
   int error = CopyTicksRange(symbol,ticks,COPY_TICKS_ALL,(ulong)From_Past*1000,(ulong)To_Present*1000+60000); //conversion from datetime to milliseconds
   if(error == -1){ printf("<!> ERROR Ticks download<!> = %d ",GetLastError()); return 0;}                                                                                   //add 60000 milliseconds to include the last minute specified in the interval
   
   MqlDateTime DateAndTime;
   string date;
   
   MyMqlRates RatesSeconde[];
   ArrayResize(RatesSeconde,(count)*6,0);
   
   long sommeTicks = 0;
   double high;
   double low;
   int ticks_closeHigh;
   int ticks_closeLow;
   
   for(int nb_candle_M1=0; nb_candle_M1<count;nb_candle_M1++){    //Nb of M1 candles copied into "Rates"
      date = TimeToString(ticks[sommeTicks].time,TIME_DATE);
      TimeToStruct(ticks[sommeTicks].time,DateAndTime);
      
      for(int Sec10=0; Sec10<=5; Sec10++){                        //Nb of Sec10 candles that will be created in RatesSeconde
         high = 0;
         low = ticks[sommeTicks].bid;
         ticks_closeHigh = 0;
         ticks_closeLow = 0;
         
         if(o) RatesSeconde[nb_candle_M1*6+Sec10].open = ticks[sommeTicks].bid;                  //allocation of the opening price value because the ticks are sorted from the oldest to the most recent 
         if(c) RatesSeconde[nb_candle_M1*6+Sec10].close = ticks[sommeTicks].bid;                 //allocation of the value of the closing price in case there were no ticks in the ten in process
         if(h) RatesSeconde[nb_candle_M1*6+Sec10].high = ticks[sommeTicks].bid;                                      //same
         if(l) RatesSeconde[nb_candle_M1*6+Sec10].low = ticks[sommeTicks].bid;                                       //same
         if(tv) RatesSeconde[nb_candle_M1*6+Sec10].tick_volume = 0;                                                   //same
         if(s) RatesSeconde[nb_candle_M1*6+Sec10].spread = ticks[sommeTicks].ask - ticks[sommeTicks].bid;            //same
         if(t) RatesSeconde[nb_candle_M1*6+Sec10].time = StringToTime(date + " " + (string)DateAndTime.hour + ":" + (string)DateAndTime.min + ":" + (string)(Sec10*10)); //same & adding second at time data
         if(tch) RatesSeconde[nb_candle_M1*6+Sec10].ticks_closeHigh = 0;
         if(tcl) RatesSeconde[nb_candle_M1*6+Sec10].ticks_closeLow = 0;
         if(vco) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeOpen = 0;
         if(vch) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeHigh = 0;
         if(vcl) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeLow = 0;
         if(ap) RatesSeconde[nb_candle_M1*6+Sec10].average_price = ticks[sommeTicks].bid;
         
         if(DateAndTime.sec>=Sec10*10 && DateAndTime.sec<Sec10*10+10){                            //if the ticks are in the range of the RatesSecond during processing
            
            long i;
            for(i=sommeTicks; DateAndTime.sec>=Sec10*10 && DateAndTime.sec<Sec10*10+10; i++){          //course of the ticks of the ten in process
               if(ticks[i].bid>high)
               {
                  high = ticks[i].bid;
                  ticks_closeHigh = (int)(i-sommeTicks);
               }
               if(ticks[i].bid<low)
               {
                  low = ticks[i].bid;
                  ticks_closeLow = (int)(i-sommeTicks);
               }
               if((i+1)==ArraySize(ticks)) break;
               TimeToStruct(ticks[i+1].time, DateAndTime);
            }
            
            if(tv) RatesSeconde[nb_candle_M1*6+Sec10].tick_volume = (int)(i-sommeTicks);                             //Ticks_volume
            sommeTicks = i;
            if(c) RatesSeconde[nb_candle_M1*6+Sec10].close = ticks[sommeTicks].bid;                          //Close
            if(h) RatesSeconde[nb_candle_M1*6+Sec10].high = high;                                            //High
            if(l) RatesSeconde[nb_candle_M1*6+Sec10].low = low;                                              //Low
            if(s) RatesSeconde[nb_candle_M1*6+Sec10].spread = ticks[sommeTicks].ask-ticks[sommeTicks].bid;   //spread
            if(tch) RatesSeconde[nb_candle_M1*6+Sec10].ticks_closeHigh = ticks_closeHigh;
            if(tcl) RatesSeconde[nb_candle_M1*6+Sec10].ticks_closeLow = ticks_closeLow;
            if(vco) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeOpen = 100*(RatesSeconde[nb_candle_M1*6+Sec10].close - RatesSeconde[nb_candle_M1*6+Sec10].open)/RatesSeconde[nb_candle_M1*6+Sec10].open;
            if(vch) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeHigh = 100*(RatesSeconde[nb_candle_M1*6+Sec10].close - RatesSeconde[nb_candle_M1*6+Sec10].high)/RatesSeconde[nb_candle_M1*6+Sec10].high;
            if(vcl) RatesSeconde[nb_candle_M1*6+Sec10].variation_closeLow = 100*(RatesSeconde[nb_candle_M1*6+Sec10].close - RatesSeconde[nb_candle_M1*6+Sec10].low)/RatesSeconde[nb_candle_M1*6+Sec10].low;
            if(ap) RatesSeconde[nb_candle_M1*6+Sec10].average_price = (RatesSeconde[nb_candle_M1*6+Sec10].close + RatesSeconde[nb_candle_M1*6+Sec10].open + high + low)/4.0;
         }         
      }
   }
   
   if(tv) getVolumeProfile(RatesSeconde);
   
   ArrayCopy(Rates_Seconde,RatesSeconde,0,0,WHOLE_ARRAY);
   ArrayReverse(Rates_Seconde,0,WHOLE_ARRAY);  
   
   return count;
}

//+--------------------------------------------------------------------------------------------------------------+

void getVolumeProfile(MyMqlRates &RatesSeconde[])
{
   double lowest=RatesSeconde[0].low;
   double highest=RatesSeconde[0].high;
   for(int k=1; k<ArraySize(RatesSeconde); k++)                 //search for the lowest price in the RatesSeconde list.
   {
      if(RatesSeconde[k].low<lowest) lowest = RatesSeconde[k].low;
      if(RatesSeconde[k].high>highest) highest = RatesSeconde[k].high;
   }
   
   while((int)MathLog10((int)lowest)+1 != 5) lowest *=10;       //definition of the reference price to calculate the index of the volume_profile list with a precision of 5 digits for the price.
   int reference_price_index = (int)lowest;
   
   while((int)MathLog10((int)highest)+1 != 5) highest *=10;       //definition of the reference price to calculate the index of the volume_profile list with a precision of 5 digits for the price.
   int reference_price_index_high = (int)highest+1;
   
   double price;
   int volume_profile[];
   ArrayResize(volume_profile,reference_price_index_high-reference_price_index,1);
   ArrayInitialize(volume_profile, 0);
   
   for(int j=0; j<ArraySize(RatesSeconde); j++)
   {
      price = RatesSeconde[j].average_price;
      while((int)MathLog10((int)price)+1 != 5) price *=10;      //normalization of the price to access the corresponding index in the volume_profile list.
      volume_profile[(int)price - reference_price_index] +=RatesSeconde[j].tick_volume;   //incrementing the volume corresponding to the price : "price".
   }
   
   for(int i=0;i<ArraySize(RatesSeconde); i++)
   {
      price = RatesSeconde[i].average_price;
      while((int)MathLog10((int)price)+1 != 5) price *=10;      //normalization of the price to access the corresponding index in the volume_profile list.
      RatesSeconde[i].volume_profile += (int)volume_profile[(int)price - reference_price_index];   //incrementing the volume corresponding to the price : "price".
   }
}

//+--------------------------------------------------------------------------------------------------------------+
//+--------------------------------------------------------------------------------------------------------------+