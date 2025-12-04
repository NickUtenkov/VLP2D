using System;
using System.IO;
using System.Text.Json;

namespace VLP2D.Common
{
	internal class UtilsJson
	{
		public static T jsonDeserialize<T>(string path)
		{
			T rc = default(T);
			try
			{
				using (StreamReader reader = new StreamReader(path))
				{
					string linesAll = "";
					string line;
					while ((line = reader.ReadLine()) != null)
					{
						linesAll += line;
					}
					rc = JsonSerializer.Deserialize<T>(linesAll, new JsonSerializerOptions { IncludeFields = true });
				}
			}
			catch (Exception)
			{
			}
			return rc;
		}

		public static void jsonSerialize<T>(T obj, string path)
		{
			try
			{
				using (StreamWriter writer = new StreamWriter(path))
				{
					string jsonString = JsonSerializer.Serialize(obj, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true });
					writer.WriteLine(jsonString);
				}
			}
			catch (Exception)
			{
			}
		}
	}
}
