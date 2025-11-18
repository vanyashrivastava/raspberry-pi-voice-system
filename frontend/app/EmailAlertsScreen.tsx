import React from "react";
import { View, Text, StyleSheet, FlatList } from "react-native";
import { colors } from "../theme/colors";
import AlertCard from "./components/AlertCard";
import SectionHeader from "./components/SectionHeader";
import StatsBar from "./components/StatsBar";

const mockEmails = [
  {
    id: "1",
    title: "Amazon Security Alert",
    sender: "security@amazon.com",
    subject: "Unusual login attempt",
    preview: "We detected an unusual login attempt on your account. Click here to verify...",
    risk: "High",
    timestamp: "2 min ago",
  },
  {
    id: "2",
    title: "Bank of America",
    sender: "alerts@bankofamerica.com",
    subject: "Confirm your identity",
    preview: "Please confirm your identity to continue using your account securely.",
    risk: "High",
    timestamp: "15 min ago",
  },
  {
    id: "3",
    title: "PayPal Account",
    sender: "service@paypal.com",
    subject: "Action required on your account",
    preview: "Your account has been locked due to suspicious activity.",
    risk: "Medium",
    timestamp: "1 hour ago",
  },
];

export default function EmailAlertsScreen({ navigation }) {
  const stats = [
    { value: 3, label: "High Risk" },
    { value: 12, label: "Total Alerts" },
    { value: 42, label: "Residents" },
  ];

  return (
    <View style={styles.container}>
      <SectionHeader
        title="Email Alerts"
        subtitle="Recent suspicious emails detected"
      />
      <StatsBar stats={stats} />

      <FlatList
        data={mockEmails}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <AlertCard
            item={item}
            onPress={() => navigation.navigate("EmailDetails", { email: item })}
          />
        )}
        scrollEnabled={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
});
