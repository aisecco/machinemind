select TRADEDATE as `date`, TOPEN as  `open`,TCLOSE as `close`,THIGH as `high`, TLOW as `low`, VOL / 100 as `volume`, '600519' as `code` from tq_qt_skdailyprice_m 
where (SECODE = '2010000438' or SENAME like '%茅台%' )
and TRADEDATE >= '20200424'
order by TRADEDATE