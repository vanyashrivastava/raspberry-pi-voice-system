import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import LandingScreen from './app/LandingScreen';
import HomeScreen from './app/HomeScreen';
import AlertDetailsScreen from './app/AlertDetailsScreen';
import EmailAlertsScreen from './app/EmailAlertsScreen';
import EmailDetailsScreen from './app/EmailDetailsScreen';
import CallMonitoringScreen from './app/CallMonitoringScreen';
import ResidentListScreen from './app/ResidentListScreen';
import ResidentProfileScreen from './app/ResidentProfileScreen';
import SettingsScreen from './app/SettingsScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Landing" component={LandingScreen} />
        <Stack.Screen name="Home" component={HomeScreen} />
        <Stack.Screen name="AlertDetails" component={AlertDetailsScreen} />
        <Stack.Screen name="EmailAlerts" component={EmailAlertsScreen} />
        <Stack.Screen name="EmailDetails" component={EmailDetailsScreen} />
        <Stack.Screen name="CallMonitoring" component={CallMonitoringScreen} />
        <Stack.Screen name="ResidentList" component={ResidentListScreen} />
        <Stack.Screen name="ResidentProfile" component={ResidentProfileScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
