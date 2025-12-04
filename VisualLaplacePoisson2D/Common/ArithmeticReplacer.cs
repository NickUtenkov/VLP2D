using MathSubstitutor;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace VLP2D.Common
{
	public static class ArithmeticReplacer
	{
		struct MatchInfo
		{
			public int Index { get; set; }
			public int Length { get; set; }
			public string Value { get; set; }
			public MatchInfo(int index, int length, string value)
			{
				Index = index;
				Length = length;
				Value = value;

			}
		};

		public static string replaceArithmeticOperators(string str)
		{
			List<string> matches = findFunctionsAndArrayElements(str);
			(string, Dictionary<string, string>) replaced = replace(str, matches);
			string strRes = Substitutor.replaceArithmeticOperators(replaced.Item1);
			foreach (KeyValuePair<string, string> kv in replaced.Item2)
			{
				strRes = strRes.Replace(kv.Key, kv.Value);
			}

			return strRes;

			static List<string> findFunctionsAndArrayElements(string str)
			{
				Regex regex1 = new Regex(@"\w+\(([^)]*)\)");
				Regex regex2 = new Regex(@"\w+\[([^]]*)\]");
				MatchCollection matches1 = regex1.Matches(str);
				MatchCollection matches2 = regex2.Matches(str);
				List<string> matches = new List<string>();
				List<MatchInfo> matchInfos1 = new List<MatchInfo>();
				List<MatchInfo> matchInfos2 = new List<MatchInfo>();
				matchesToList(matches1, matchInfos1);
				matchesToList(matches2, matchInfos2);

				balanceAllBrackets(str, matchInfos1, '(', ')');
				balanceAllBrackets(str, matchInfos2, '[', ']');

				foreach (MatchInfo match1 in matchInfos1)
				{
					bool isIn2 = false;
					foreach (MatchInfo match2 in matchInfos2)
					{
						if (match1.Index > match2.Index && match1.Index < match2.Index + match2.Length)
						{
							isIn2 = true;
							break;
						}
					}
					if (!isIn2) matches.Add(match1.Value);
				}
				foreach (MatchInfo match2 in matchInfos2)
				{
					bool isIn1 = false;
					foreach (MatchInfo match1 in matchInfos1)
					{
						if (match2.Index > match1.Index && match2.Index < match1.Index + match1.Length)
						{
							isIn1 = true;
							break;
						}
					}
					if (!isIn1) matches.Add(match2.Value);
				}

				return matches;
			}

			static (string, Dictionary<string, string>) replace(string str, List<string> matches)
			{
				Dictionary<string, string> dict = new Dictionary<string, string>();
				int count = 0;
				foreach (string match in matches)
				{
					var key = "xyz" + count;
					dict.Add(key, match);
					str = str.Replace(match, key);
					count++;
				}
				return (str, dict);
			}

			static void matchesToList(MatchCollection matches, List<MatchInfo> list)
			{
				foreach (Match match in matches) list.Add(new MatchInfo(match.Index, match.Length, match.Value));
			}

			static void balanceMatchBrackets(string str, char brOpen, char brClose, ref MatchInfo info)
			{
				int cOpen = info.Value.Count(x => x == brOpen);
				int cClose = info.Value.Count(x => x == brClose);
				if (cOpen == cClose) return;
				cOpen = 0;
				cClose = 0;
				for (int i = info.Index; i < str.Length; i++)
				{
					char ch = str[i];
					if (ch == brOpen) cOpen++;
					if (ch == brClose) cClose++;
					if (cOpen > 0 && cOpen == cClose)
					{
						info.Length = i - info.Index + 1;
						info.Value = str.Substring(info.Index, info.Length);
						return;
					}
				}
			}

			static void balanceAllBrackets(string str, List<MatchInfo> matchInfos, char brOpen, char brClose)
			{
				for (int i = 0; i < matchInfos.Count; i++)
				{
					MatchInfo info = matchInfos[i];
					balanceMatchBrackets(str, brOpen, brClose, ref info);
					matchInfos[i] = info;
				}
			}
		}

		public static string replaceHPMacros(string str)
		{
			int idxStart = 0;
			while (true)
			{
				(int, int) posLen = getHPMacro(str, idxStart);
				if (posLen.Item1 == -1) break;
				string strMacro = str.Substring(posLen.Item1, posLen.Item2);
				string strExpression = str.Substring(posLen.Item1 + 3, posLen.Item2 - 4);
				string strReplaced = replaceArithmeticOperators(strExpression);
				str = str.Replace(strMacro, strReplaced);
				idxStart = posLen.Item1 + posLen.Item2;
			}
			return str;

			static (int, int) getHPMacro(string str, int idxStart)
			{
				int idx = str.IndexOf("HP(");
				if (idx != -1)
				{
					int countOpen = 1, countClose = 0;
					for (int i = idx + 3; i < str.Length; i++)
					{
						char ch = str[i];
						if (ch == '(') countOpen++;
						if (ch == ')') countClose++;
						if (countOpen == countClose) return (idx, i - idx + 1);
					}
				}
				return (-1, 0);
			}
		}

		public static string convertTo_mul_HD<T>(string hpVar, string intVar)
		{
			string strType = typeof(T).Name;

			if (strType == "Single" || strType == "Double") return string.Format("{0} * {1}", hpVar, intVar);
			if (strType == "DD128" || strType == "QD256") return string.Format("mul_HD({0}, {1})", hpVar, intVar);

			return "";
		}
	}
}
